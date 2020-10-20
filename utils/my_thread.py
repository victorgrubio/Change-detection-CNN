#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:29:09 2019

@author: visiona
"""
from threading import Thread, Event, Lock


class MyThread(Thread):

    def __init__(self, logger, target, args):
        Thread.__init__(self, target=target, args=args)
        self.logger = logger
        self.is_running_lock = Lock()
        self.is_running = True
        self.is_stopped_lock = Lock()
        self._stop_event = Event()

    def get_stopped(self):
        self.is_stopped_lock.acquire(True)
        val = self._stop_event.is_set()
        self.is_stopped_lock.release()
        return val

    def set_stopped(self):
        self.is_stopped_lock.acquire(True)
        self._stop_event.set()
        self.is_stopped_lock.release()

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
