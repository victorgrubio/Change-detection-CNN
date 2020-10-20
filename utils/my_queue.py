# -*- coding: utf-8 -*-
from queue import Queue
from threading import Lock
from utils.my_thread import MyThread
import time


class MyQueue(Queue):

    def __init__(self, logger, size):
        Queue.__init__(self, maxsize=size)
        self.logger = logger
        self.is_stopped_lock = Lock()
        self.is_stopped = False
        self.is_running_lock = Lock()
        self.is_running = True
        self.thread = None

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
    def is_stopped(self):
        self.is_stopped_lock.acquire(True)
        val = self.__is_stopped
        self.is_stopped_lock.release()
        return val

    @is_stopped.setter
    def is_stopped(self, val):
        self.is_stopped_lock.acquire(True)
        self.__is_stopped = val
        self.is_stopped_lock.release()

    # start a thread that calls run method
    def start(self):
        thread = MyThread(self.logger, target=self.run, args=())
        thread.daemon = False
        self.thread = thread
        thread.start()
        return self

    # Finalize the current thread
    def finalize(self):
        self.thread.is_running = False

    # Target method for threads
    def run(self):
        pass

    # Blocking call: return True when there is one or more frames to detect
    def more(self):
        while self.empty():  # Block until update() yields one frame
            time.sleep(0.005)
        return True

    def clear(self):
        self.queue.clear()
