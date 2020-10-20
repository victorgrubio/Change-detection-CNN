import cv2
import time
from utils.my_queue import MyQueue


class VideoQueue(MyQueue):
    # class to read streams using threads

    def __init__(self, logger, queue_size=1, path=0, fps=30, has_buffer=False):
        MyQueue.__init__(self, logger, queue_size)
        self.cap = self.get_video_cap(path)
        self.path = path
        self.fps = fps
        self.has_buffer = has_buffer

    def get_video_cap(self, path):
        # check if video capture works
        cap = cv2.VideoCapture(path)
        if cap.isOpened() is False:
            raise FileNotFoundError(
                'Video stream or file not found at {}'.format(path))
        else:
            return cap

    def get_drop_rate(self, cap_fps):
        """Method documentation"""
        drop_rate = 0
        if cap_fps % self.fps != 0:
            self.fps = cap_fps
            self.logger.info(
                'FPS adjusted to {}'.format(cap_fps))
        else:
            drop_rate = int(cap_fps / self.fps)
            self.logger.debug('Droprate: {}'.format(drop_rate))
        return drop_rate

    def run(self):
        cap_fps = int(self.cap.get(5) + 1)
        last_read_frame_time = 0
        last_added_frame_time = 0
        counter_frames = 0
        drop_rate = self.get_drop_rate(cap_fps)
        timespace_frames = 1 / cap_fps
        while self.thread.is_running:
            if (time.time() - last_read_frame_time) > timespace_frames:
                if not self.full():
                    # read current frame from capture
                    ret, frame = self.cap.read()
                    counter_frames += 1
                    last_read_frame_time = time.time()
                    if not ret:
                        self.logger.warn('Thread has stopped due to not ret')
                        self.is_stopped = True
                    if counter_frames == drop_rate:
                        #self.logger.debug('Added frame to queue')
                        self.logger.log(
                            0, 'Last frame was added {} s ago'.format(
                            time.time() - last_added_frame_time))
                        last_added_frame_time = time.time()
                        self.put(frame)
                        self.logger.log(0, 'Current size of queue: {}'.format(
                            self.qsize()))
                        counter_frames = 0
                # If the queue is full, clear it and reset. Only for streams
                elif not self.has_buffer:
                    self.clear()
                    self.logger.debug(
                        'Queue reached maximum capacity '
                        '({}). Cleared'.format(self.maxsize))
            else:
                #self.logger.debug('SLEEPING for {:.3f}'.format(
                #   timespace_frames))
                time.sleep(timespace_frames)
        self.logger.info('videoThread not running')
        self.thread.is_running = False
        self.is_running = False
