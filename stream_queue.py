import cv2
import time
from utils.my_queue import MyQueue


class StreamQueue(MyQueue):
    # class to read streams using threads

    def __init__(self, logger, queue_size=1, path=0, fps=30):
        MyQueue.__init__(self, logger, queue_size)
        self.cap = self.get_video_cap(path)
        self.path = path
        self.fps = fps

    def get_video_cap(self, path):
        # check if video capture works
        cap = cv2.VideoCapture(path)
        if cap.isOpened() is False:
            raise FileNotFoundError(
                'Video stream or file not found at {}'.format(path))
        else:
            return cap

    def run(self):
        last_fps_time = time.time()
        timespace_frames = 1 / self.fps
        while self.thread.is_running:
            if (time.time() - last_fps_time) > timespace_frames:
                if not self.full():
                    # read current frame from capture
                    ret, frame = self.cap.read()
                    last_fps_time = time.time()
                    if not ret:
                        self.logger.warn('Thread has stopped due to not ret')
                        self.is_stopped = True
                    self.logger.debug('Added frame to queue')
                    self.put(frame)
                    self.logger.debug('Current size of queue: {}'.format(
                        self.qsize()))
                    self.logger.debug('SLEEPING for {:.2f}'.format(
                        timespace_frames))
                # If the queue is full, clear it and reset. Only for streams
                else:
                    self.clear()
                    self.logger.debug('Queue reached maximum capacity ({}).\
                                      Cleared'.format(self.maxsize))
            else:
                time.sleep(timespace_frames)
        self.logger.info('videoThread not running')
        self.thread.is_running = False
        self.is_running = False
