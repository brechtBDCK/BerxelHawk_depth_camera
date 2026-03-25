#coding=utf-8
import sys
import os
import time
import datetime
import threading
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('../BerxelSdkDriver/')

from BerxelSdkDriver.BerxelHawkContext import *
from BerxelSdkDriver.BerxelHawkDevice import *
from BerxelSdkDriver.BerxelHawkFrame import *
from BerxelSdkDriver.BerxelHawkDefines import *



class HawkColor(object):

    def __init__(self):
        self.__context = None
        self.__device = None
        self.__device2 = None
        self.__deviceList = []

    # Step 1 open device
    def openDevice(self):


        self.__context = BerxelHawkContext()

        if self.__context is None:
            print("init failed")
            return  False

        self.__context.initCamera()

        self.__deviceList = self.__context.getDeviceList()

        if len(self.__deviceList) <  2:
            print("not have two device， can not double openDevie")
            return False

        print("device addres1 : ", self.__deviceList[0].deviceAddress)
        self.__device = self.__context.openDevice(self.__deviceList[0])

        if self.__device is None:
            return False

        print("open device1 succed")

        print("device addres2 : ", self.__deviceList[1].deviceAddress)
        self.__device2 = self.__context.openDevice(self.__deviceList[1])

        if self.__device2 is None:
            print("can not open secend device")
            return False
        print("open device2 succed")

        return True

    # Step 2 ： open Stream

    def startStream(self):

        if self.__device is None:
            return  False

        frameMode = self.__device.getCurrentFrameMode(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])
        self.__device.setFrameMode(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'] ,frameMode)

        ret = self.__device.startStreams(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])

        ret1 =  self.__device2.startStreams(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])

        if ret == 0:
            print("start stream1 succeed")
        else:
            print("start stream1 failed")
            return False

        if ret1 == 0:
            print("start stream2 succeed")
        else:
            print("start stream2 failed")
            return False

        return True
    # 3: read Frame

    def displayImage(self):

        hawkFrame = self.__device.readColorFrame(30)
        if hawkFrame is None:
            return 1

        width  = hawkFrame.getWidth()
        height = hawkFrame.getHeight()
        streamType = hawkFrame.getStreamType()
        dataSize = hawkFrame.getDataSize()
        pxielType = hawkFrame.getPixelType()
        index = hawkFrame.getFrameIndex()
        frameBuffer = hawkFrame.getDataAsUint8()


        color_array = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=frameBuffer)
        img = cv2.cvtColor(np.uint8(color_array), cv2.COLOR_BGR2RGB)

        cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color', img)



        ch = 0xFF & cv2.waitKey(1)

        if ch == 27 or ch == 81 or ch == 113:
            return -1

        self.__device.releaseFrame(hawkFrame)

        return 1

    def displayImage2(self):

        hawkFrame = self.__device2.readColorFrame(30)
        if hawkFrame is None:
            return 1

        width  = hawkFrame.getWidth()
        height = hawkFrame.getHeight()
        streamType = hawkFrame.getStreamType()
        dataSize = hawkFrame.getDataSize()
        pxielType = hawkFrame.getPixelType()
        index = hawkFrame.getFrameIndex()
        frameBuffer = hawkFrame.getDataAsUint8()


        color_array = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=frameBuffer)
        img = cv2.cvtColor(np.uint8(color_array), cv2.COLOR_BGR2RGB)

        cv2.namedWindow("Color2", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color2', img)



        ch = 0xFF & cv2.waitKey(1)

        if ch == 27 or ch == 81 or ch == 113:
            return -1

        self.__device2.releaseFrame(hawkFrame)

        return 1

    def ShowFrame(self):

        print ("Sart Show Frame...")

        time.sleep(1)
        while 1:
            ret = self.displayImage()
            if ret != 1:
                break

        self.closeStream()
        self.closeDevice()
        return

    def ShowFrame2(self):

        print("Sart Show Frame...")

        time.sleep(1)
        while 1:
            ret = self.displayImage2()
            if ret != 1:
                break

        self.closeStream()
        self.closeDevice()
        return



    #4 : closeStream
    def closeStream(self):
        if self.__device is None:
            return  False

        ret =  self.__device.stopStream(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])
        ret2 = self.__device2.stopStream(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])
        if ret == 0:
            print("close stream1 succeed")
          #  return True
        else:
            print("clsoe stream1 failed")
            return False
        if ret2 == 0:
            print("close stream2 succeed")
            #  return True
        else:
            print("clsoe stream2 failed")
            return False

        return True

    #5：close Device
    def closeDevice(self):
        if self.__context is None:
            return False

        if self.__device is None:
            return  False

        ret = self.__context.closeDevice(self.__device)

        ret = self.__context.closeDevice(self.__device2)

        if ret == 0:
            print("clsoe device succeed")
        else:
            print("close device Failed")

        self.__context.destroyCamera()

    def StartTest(self):

        if self.openDevice() == False:
            return
        #
        if self.startStream() == False:
            return

        tShowFrame = threading.Thread(target=self.ShowFrame)
        tShowFrame2 = threading.Thread(target=self.ShowFrame2)
        tShowFrame.start()
        tShowFrame2.start()




if __name__ == '__main__':
    print('PyCharm')
    colorView = HawkColor()
    colorView.StartTest()
