#!/usr/bin/python
#coding=utf-8


try:
    from .BerxelHawkDefines import  *
except ImportError:
    from BerxelHawkDefines import  *
from ctypes import *
import os
import time
import datetime
import threading
import sys

class BerxelHawkFrame(object):

    def __init__(self, frame_Handle = None):
        self._frameHandle = frame_Handle


    def getDataAsUint8(self):
        if self._frameHandle is None:
            return None
        return (c_uint8 * int(self._frameHandle.contents.dataSize)).from_address(self._frameHandle.contents.pVoidData)

    def getOriData(self):
        if self._frameHandle is None:
            return None
        return self._frameHandle.contents.pVoidData

    def getDataAsUint16(self):
        if self._frameHandle is None:
            return None
        return (c_uint16 * int(self._frameHandle.contents.dataSize /2)).from_address(self._frameHandle.contents.pVoidData)

    def getStreamType(self):
        if self._frameHandle is None:
            return 0
        return self._frameHandle.contents.type

    def getFrameIndex(self):
        if self._frameHandle is None:
            return 0
        return  self._frameHandle.contents.frameIndex

    def getPixelType(self):
        if self._frameHandle is None:
            return 0
        return self._frameHandle.contents.pixelType

    def getWidth(self):
        if self._frameHandle is None:
            return 0
        return self._frameHandle.contents.width

    def getHeight(self):
        if self._frameHandle is None:
            return 0
        return  self._frameHandle.contents.height

    def getDataSize(self):
        if self._frameHandle is None:
            return 0
        return self._frameHandle.contents.dataSize

    def getTimeStamp(self):
        if self._frameHandle is None:
            return 0
        return self._frameHandle.contents.timestamp

    def getFrameHandle(self):
        if self._frameHandle is None:
            return None
        return self._frameHandle

    def getFps(self):
        if self._frameHandle is None:
            return None
        return  self._frameHandle.contents.fps
