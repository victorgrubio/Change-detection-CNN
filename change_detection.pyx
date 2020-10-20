# -*- coding: utf-8 -*-
# @Author: Victor Garcia
# @Date:   2018-09-25 09:35:30
# @Last Modified by:   Victor Garcia
# @Last Modified time: 2018-09-25 09:35:30
import numpy as np
cimport numpy as np
import yaml
import os
import argparse
import sys
import cv2
import time
import traceback
from datetime import datetime
from threading import Lock

# files with auxiliary methods
import pyximport
pyximport.install()
from utils.helpers import parse_input
from utils import img_preprocessing as img_prep
from utils import model_preprocessing as model_prep
from video_queue import VideoQueue
from utils.img_preprocessing import Aligner


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ChangeDetection():

    def __init__(self, logger, args, model):
        self.logger = logger
        self.model = model
        self.config = self.load_config('cfg/detector.yaml', args)
        self.img_dict = {'predict_img_thresh': None, 'predict_img': None,
                         'back_img': None, 'fore_img': None,
                         'change_img': None, 'fore_aligned_img': None,
                         'boxes_img': None}
        self.img_cuts_dict = {
            'cut_top': max(self.config['img']['cut_top'], 0),
            'cut_bot': -max(self.config['img']['cut_bot'], 0),
            'cut_left': max(self.config['img']['cut_left'], 0),
            'cut_right': -max(self.config['img']['cut_right'], 0)
        }
        self.aligner = None
        self.img_shape = None
        self.video_fore_queue = None
        self.video_back_queue = None
        self.videos = {}
        self.is_resized = False
        self.is_running_lock = Lock()
        self.is_running = True
        self.is_stopping_lock = Lock()
        self.is_stopping = False

    @property
    def is_running(self):
        self.is_running_lock.acquire(True)
        val = self.__is_running
        self.is_running_lock.release()
        return val

    @is_running.setter
    def is_running(self, value):
        self.is_running_lock.acquire(True)
        self.__is_running = value
        self.is_running_lock.release()

    @property
    def is_stopping(self):
        self.is_stopping_lock.acquire(True)
        val = self.__is_stopping
        self.is_stopping_lock.release()
        return val

    @is_stopping.setter
    def is_stopping(self, value):
        self.is_stopping_lock.acquire(True)
        self.__is_stopping = value
        self.is_stopping_lock.release()

    def load_config(self, config_file, args=None):
        """
        Load config file
        """
        config = ""
        if os.path.exists(config_file):
            with open(config_file, 'rt') as f:
                config = yaml.safe_load(f.read())
        if args is not None:
            # Remove none elements
            args = {k: v for k, v in args.items() if v is not None}
            config['args'].update(args)
        self.logger.debug('Config {}'.format(config))
        return config

    def setup_video_detector(self, fore_video_path, back_video_path):
        """Method documentation"""
        video_path_dict = self.get_paths(fore_video_path, back_video_path)
        self.video_fore_queue = VideoQueue(
            self.logger, queue_size=self.config['video_queues']['queue_size'],
            path=video_path_dict['fore_path'], fps=self.config['video_queues']['fps'])
        self.video_back_queue = VideoQueue(
            self.logger, queue_size=self.config['video_queues']['queue_size'],
            path=video_path_dict['back_path'], fps=self.config['video_queues']['fps'])
        ret, tmp_img = self.video_fore_queue.cap.read()
        self.width, self.height = img_prep.get_img_dims(
            tmp_img, self.config['img']['max_width'], self.config['img']['max_width'],
            self.config['img']['model_input'])
        if self.width != tmp_img.shape[1] or self.height != tmp_img.shape[0]:
            self.is_resized = True
        # create the windows needed for img display
        if self.config['args']['align']:
            self.aligner = Aligner()
        if self.config['args']['display']:
            img_prep.create_windows(self.img_dict)
        # First frame creates video file
        if self.config['args']['save_results']:
            self.create_videos(self.config['record']['video_name_list'])
        self.video_fore_queue.start()
        self.video_back_queue.start()

    def predict_video(self, fore_video_path, back_video_path):
        """
        Method for predicting video
        """
        self.setup_video_detector(fore_video_path, back_video_path)
        while self.is_running:
            total_time = time.time()
            if self.video_fore_queue.empty() and self.video_back_queue.empty():
                time.sleep(0.005)
                continue
            try:
                if self.video_fore_queue and self.video_back_queue:
                    fore_img = self.video_fore_queue.get()
                    back_img = self.video_back_queue.get()
                    if type(fore_img) is np.ndarray and \
                            type(back_img) is np.ndarray:
                        start_time = time.time()
                        if self.is_resized:
                            fore_img = cv2.resize(fore_img, (self.width, self.height),
                                         interpolation=cv2.INTER_CUBIC)
                            back_img = cv2.resize(back_img, (self.width, self.height),
                                         interpolation=cv2.INTER_CUBIC)
                        self.detect_changes(fore_img, back_img)
                        # Third frame checks if sizes are ok.
                        # First and second ones acquired previously
                        if int(self.video_fore_queue.cap.get(1)) in [5, 6]:
                            for name in self.config['record']['video_name_list']:
                                self.logger.info('{} shape: {}'.format(
                                    name, self.img_dict[name].shape))
                            # Write frame on video
                        if self.config['args']['save_results']:
                            self.write_videos(self.config['record']['video_name_list'])
                        if self.config['args']['display']:
                            k = 0
                            for name, img in list(self.img_dict.items()):
                                cv2.imshow(name, img)
                            k = cv2.waitKey(1)
                            if k == 27:
                                raise KeyboardInterrupt(
                                    'User has stopped the program by pressing '
                                    'ESC')
                        self.logger.info('Total time on img {}'.format(time.time() - total_time))
                    else:
                        self.logger.info('One or both queues have stopped')
                        self.finalize_video_queues()
                        break
            except KeyboardInterrupt:
                self.logger.info('User interrupted the program')
                if self.config['args']['display']:
                    cv2.destroyAllWindows()
                self.finalize()
            except:
                self.logger.error('Unexpected error : {}'.format(traceback.format_exc()))
                if self.config['args']['display']:
                    cv2.destroyAllWindows()
                self.finalize()

    def detect_changes(self, fore_img, back_img):
        """
        Change detection process on imgs
        """
        self.img_dict['fore_img'] = fore_img
        if self.config['args']['align']:
            start_align_time = time.time()
            fore_img, h = img_prep.align_imgs(self.aligner, fore_img, back_img)
            self.logger.debug('Alignment time: {} s'.format(
                time.time() - start_align_time))
            fore_img = img_prep.cut_img(fore_img, self.img_cuts_dict)
            back_img = img_prep.cut_img(back_img, self.img_cuts_dict)
        self.img_dict['fore_aligned_img'] = fore_img.copy()
        self.img_processing(fore_img, back_img)

    def img_processing(self, fore_img, back_img):
        """
        Main processing method. Applies the patch predicts, updates them
        and obtains the predict img along with the other imgs.
        """
        # set overlapping fraction of each predicted window
        # set lower value of gray img to be detected as a change.
        # Strict: 150 - 200 (BINARY)
        cdef int step = self.config['img']['step']
        cdef(int, int) gray_img_shape = (fore_img.shape[0], fore_img.shape[1])
        cdef(int, int) patch_shape = (
            self.config['img']['model_input'], self.config['img']['model_input'])
        start_time = time.time()
        self.logger.log(0, "New predict has started, please wait ...")
        # get the array of predicts using a sliding window algorithm
        # the mode must be specified, please check utils.py
        # for different mode performance
        predicts_array = model_prep.patch_predicts(
            fore_img, back_img, self.model, step, self.config['args']['debug'])
        predicts_array = model_prep.array_to_patch(
            predicts_array, patch_shape, self.config['args']['debug'])
        predict_img = model_prep.get_predict_img(
            gray_img_shape, predicts_array, self.config['img']['model_input'], step)
        self.post_process_img(fore_img, back_img, predict_img)
        self.logger.debug("predict time: {}(s)".format(
            str(time.time() - start_time)))

    def post_process_img(self, fore_img, back_img, predict_img):
        """Method documentation"""

        # use THRESH_TOZERO for gradual, THRESH_BINARY to white/black
        cdef int lower_threshold = self.config['img']['threshold']
        ret, predict_img_thresh = cv2.threshold(
            predict_img, lower_threshold, 255, cv2.THRESH_BINARY)
        # dilation applied to fill the black zones near the object identified
        predict_img_thresh = img_prep.morph_erode(predict_img_thresh, 5)
        predict_img_thresh = img_prep.morph_dilate(predict_img_thresh, 7)
        # find the most significant contours and then draws them on fore img
        contours = img_prep.find_contours(predict_img_thresh)
        boxes_img = img_prep.draw_contours(fore_img.copy(), contours)
        predict_img_thresh = np.stack((predict_img_thresh,) * 3, -1)
        change_img = cv2.bitwise_and(fore_img, predict_img_thresh)
        predict_img = cv2.cvtColor(predict_img, cv2.COLOR_GRAY2BGR)
        img_dict = {'predict_img_thresh': predict_img_thresh,
                    'predict_img': predict_img, 'back_img': back_img,
                    'change_img': change_img, 'boxes_img': boxes_img}
        self.img_dict.update(img_dict)

    def get_paths(self, fore_path, back_path):
        """
        Method docs
        """
        path_dict = {'fore_path': fore_path, 'back_path': back_path}
        if fore_path is None:
            path_dict['fore_path'] = parse_input(input(
                "Enter foreground video path:"))
        if back_path is None:
            path_dict['back_path'] = parse_input(input(
                "Enter background video path:"))
        return path_dict

    def create_videos(self, name_list):
        """
        Method docs
        """
        date = "{:%d_%m-%H_%M}".format(datetime.now())
        ret, tmp_img = self.video_fore_queue.cap.read()
        # date as name of video, format = avi
        if self.is_resized:
            tmp_img = cv2.resize(tmp_img, (self.width, self.height),
                         interpolation=cv2.INTER_CUBIC)
        tmp_img_cut = img_prep.cut_img(
            tmp_img, self.img_cuts_dict)
        self.img_shape = tuple(np.flip(tmp_img_cut.shape[:2], 0))
        for name in name_list:
            video_name = '.'.join([date + '_' + name, 'avi'])
            self.logger.info('Video shape: {}'.format(self.img_shape))
            # Parameters: frame_rate,video_dims
            if name == 'boxes_img':
                self.videos[name] = img_prep.create_video(
                    video_name, self.config['record']['fps'], self.img_shape, codec='XVID')
            else:
                self.videos[name] = img_prep.create_video(
                    video_name, self.config['record']['fps'], self.img_shape, codec='MJPG')

    def write_videos(self, name_list):
        """
        Method docs
        """
        for name in name_list:
            if name in self.videos.keys() and name in self.img_dict.keys():
                self.videos[name].write(self.img_dict[name])

    def finalize(self):
        """
        Method docs
        """
        self.close_videos()
        self.finalize_video_queues()

    def close_videos(self):
        """
        Method docs
        """
        for video_name, video in self.videos.items():
            video.release()

    def finalize_video_queues(self):
        """
        Method docs
        """
        self.close_videos()
        for video_queue in [self.video_fore_queue, self.video_back_queue]:
            if video_queue is not None:
                video_queue.is_running = False
                video_queue.finalize()
        self.is_running = False
        if self.config['args']['display']:
            cv2.destroyAllWindows()

################################
#
# IMG AND FOLDER MODES (UNUSED) 
# 
################################

    def predict_img(self, fore_img_path, back_img_path):
        """
        Method for predicting one img from model and arguments
        """
        date = "{:%d_%m-%H_%M}".format(datetime.now())
        img_path_dict = self.get_paths(fore_img_path, back_img_path)
        fore_img = cv2.imread(img_path_dict['fore_path'])
        back_img = cv2.imread(img_path_dict['back_path'])
        # create the windows needed for img display
        if self.config['args']['display']:
            img_prep.create_windows(self.img_dict)
        self.detect_changes(fore_img, back_img)
        # Store the fore img with mask in a new jpg file
        if self.config['args']['save_results']:
            cv2.imwrite(date + '.jpg', self.img_dict['change_img'])
        if self.config['args']['display']:
            k = 0
            for name, img in list(self.img_dict.items()):
                cv2.imshow(name, img)
            k = cv2.waitKey(0)
            if k == 27:
                raise KeyboardInterrupt(
                    'User has stopped the program by pressing ESC')

    def predict_folder(self, fore_folder_path, back_folder_path):
        """
        Method for predicting img folders
        """
        date = "{:%d_%m-%H_%M}".format(datetime.now())
        cdef int counter_frames = 0
        folder_path_dict = self.get_paths(fore_folder_path, back_folder_path)
        frame_rate = 20
        # create the windows needed for img display
        if self.config['args']['display']:
            img_prep.create_windows(self.img_dict)
        # First frame creates video file
        if self.config['args']['save_results']:
            tmp_img = cv2.imread("/".join([
                folder_path_dict['fore_path'],
                os.listdir(folder_path_dict['fore_path'])[0]]))
            # date as name of video, format = avi
            video_shape = tuple(np.flip(tmp_img.shape[:2], 0))
            video_name = '.'.join([date, "avi"])
            # Parameters: frame_rate,video_dims
            video = img_prep.create_video(video_name, frame_rate, video_shape)
        # imgs must be sorted by name correctly in order to work properly
        for fore_filename, back_filename in zip(
                sorted(os.listdir(folder_path_dict['fore_path'])),
                sorted(os.listdir(folder_path_dict['back_path']))):
            fore_img = cv2.imread("/".join([
                fore_folder_path, fore_filename]))
            back_img = cv2.imread("/".join([
                back_folder_path, back_filename]))
            self.img_dict = self.detect_changes(fore_img, back_img)
            # Aligns img and cuts the sides using model size
            # to avoid sliding conflicts
            counter_frames += 1
            # Write frame on video
            if self.config['args']['save_results']:
                # write fore img with detection
                video.write(self.img_dict['change_img'])
                cv2.waitKey(1)
