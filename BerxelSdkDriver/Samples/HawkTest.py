#coding=utf-8
import os
import time
import datetime
import threading
import sys

import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('../BerxelSdkDriver/')


from BerxelSdkDriver.BerxelHawkContext import *
from BerxelSdkDriver.BerxelHawkDevice import *
from BerxelSdkDriver.BerxelHawkFrame import *
from BerxelSdkDriver.BerxelHawkDefines import *



class HawkTest(object):

    def __init__(self):
        self.__context = None
        self.__device = None
        self.__deviceList = []

    # Step 1 open device
    def openDevice(self):
        self.__context = BerxelHawkContext()

        if self.__context is None:
            print("init failed")
            return  False

        self.__context.initCamera()

        self.__deviceList = self.__context.getDeviceList()

        if len(self.__deviceList) <  1:
            print("can not find device")
            return False

        print("device addres : ", self.__deviceList[0].deviceAddress)
        self.__device = self.__context.openDevice(self.__deviceList[0])

        if self.__device is None:
            return False

        return True

    # Step 2 ： open Stream

    def startStream(self):

        if self.__device is None:
            return  False
        info = self.__device.getCurrentDeviceInfo()

        print("addr = ",info.deviceAddress)
        print("sn= ", info.serialNumber)
        print("vid = ", info.vendorId)
        print("pid = ", info.productId)

        params = self.__device.getDeviceIntrinsicParams()
        print("params" , params)

        version = self.__device.getVersion()
        print("fw version", version.sdkVersion.major, version.sdkVersion.minor, version.sdkVersion.revision)
        print("version" , version)

        self.__device.setStreamMirror(True)

        self.__device.setStreamFlagMode(BerxelHawkStreamFlagMode.forward_dict['BERXEL_HAWK_SINGULAR_STREAM_FLAG_MODE'])


        list = []
        list = self.__device.getSupportFrameModes(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])
        print("list length= " ,len(list))

        for x in range(len(list)):
            print(list[x].pixelFormat,  list[x].resolutionX, " ", list[x].resolutionY, list[x].framerate)

        framemode = BerxelHawkStreamFrameMode()

        print("sizeof",sizeof(BerxelHawkStreamFrameMode))


        self.__device.setFrameMode(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'] ,list[1])

        frameMode = self.__device.getCurrentFrameMode(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])


        print("pixel=", frameMode.pixelFormat, "  x = ", frameMode.resolutionX, " y= ", frameMode.resolutionY, "fps = ",
              frameMode.frameraqte)

        ret = self.__device.startStreams(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])

        if ret == 0:
            print("start stream succeed")
            return True
        else:
            print("start stream failed")
            return False
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

        # print("pixel = ", pxielType)
        # print("type = ", streamType)
        # print("frameIndex = ", index)
        # print("fps =" , hawkFrame.getFps())
        # print("w= ", width)
        # print("h =", height)
        # print("datasize = ",dataSize)
        # print("stamp = ", hawkFrame.getTimeStamp())
        # print(frameBuffer)


        color_array = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=frameBuffer)
        img = cv2.cvtColor(np.uint8(color_array), cv2.COLOR_BGR2RGB)

        cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color', img)



        ch = 0xFF & cv2.waitKey(1)

        if ch == 27 or ch == 81 or ch == 113:
            return -1

        self.__device.releaseFrame(hawkFrame)

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




    #4 : closeStream
    def closeStream(self):
        if self.__device is None:
            return  False

        ret = self.__device.stopStream(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])

        if ret == 0:
            print("close stream succeed")
            return True
        else:
            print("clsoe stream failed")
            return False

    #5：close Device
    def closeDevice(self):
        if self.__context is None:
            return False

        if self.__device is None:
            return  False

        ret = self.__context.closeDevice(self.__device)
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
        tShowFrame.start()




if __name__ == '__main__':
    print('PyCharm')
    depthView = HawkTest()
    depthView.StartTest()
