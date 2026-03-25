#!/usr/bin/python
#coding=utf-8

# from BerxelHawkDefines import *
try:
    from .BerxelHawkNativeMethods import *
    from .BerxelHawkFrame import *
except ImportError:
    from BerxelHawkNativeMethods import *
    from BerxelHawkFrame import *

class BerxelHawkDevice(object):

    def __init__(self, deviceHandle = None):
        self._deviceHandle = deviceHandle
        self._deviceColorHandle = None
        self._mDepthStream = None
        self._mColorStream = None
        self._mIrStream = None
        self._hasColorDevice = False


    def initColorHandle(self, deviceColorHandle):
        self._hasColorDevice = True
        self._deviceColorHandle = deviceColorHandle




    def getSupportFrameModes(self, streamType):
        modeList = []
        if self._deviceHandle is None:
            return modeList

        frameMode_Handle = frameModeHandle()
        nCount = c_uint32(0)
        print("getSupportFrameModes streamType = ", streamType)

        if (streamType == BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM']):
            if (self._hasColorDevice):
                ret = berxelGetSupportStreamFrameMode(self._deviceColorHandle, streamType, byref(frameMode_Handle),
                                                      byref(nCount))
            else:
                ret = berxelGetSupportStreamFrameMode(self._deviceHandle, streamType, byref(frameMode_Handle),
                                                      byref(nCount))
        else:
            ret = berxelGetSupportStreamFrameMode(self._deviceHandle, streamType, byref(frameMode_Handle),
                                                  byref(nCount))



        if nCount.value > 0:
            for x in range(nCount.value):
                modeList.append(frameMode_Handle[x])
                print(frameMode_Handle[x].resolutionX, " ", frameMode_Handle[x].resolutionY)
        return  modeList

    def getCurrentFrameMode(self, streamType):
        if self._deviceHandle is None:
            return None

        frameModePtr = None

        if(streamType == BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM']):
            if(self._hasColorDevice):
                frameModePtr = berxelGetCurrentStreamFrameMode(self._deviceColorHandle, streamType)
            else:
                frameModePtr = berxelGetCurrentStreamFrameMode(self._deviceHandle, streamType)
        else:
            frameModePtr = berxelGetCurrentStreamFrameMode(self._deviceHandle, streamType)





        if bool(frameModePtr) is False:
            return None
        else:
            return BerxelHawkStreamFrameMode(frameModePtr.contents.pixelFormat, frameModePtr.contents.resolutionX, frameModePtr.contents.resolutionY , frameModePtr.contents.framerate)

    def setFrameMode(self, streamType, mode):
        if self._deviceHandle is None:
            return -1

        ret = 0

        if (streamType == BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM']):
            if (self._hasColorDevice):
                ret = berxelSetStreamFrameMode(self._deviceColorHandle, streamType, mode)
                print("set color Device strmeMode")
            else:
                ret = berxelSetStreamFrameMode(self._deviceHandle, streamType, mode)
        else:
            ret = berxelSetStreamFrameMode(self._deviceHandle, streamType, mode)



        print("setFrameMode , ret = ", ret )
        return ret

    def setStreamFlagMode(self, streamFlagMode):
        if self._deviceHandle is None:
            return -1

        flag = c_uint32(streamFlagMode)


        if (self._hasColorDevice):
            berxelSetStreamFlagMode(self._deviceColorHandle, flag)

        ret = berxelSetStreamFlagMode(self._deviceHandle, flag)
        return ret


    def startStreams(self, streamFlag, callback = None, user = None):
        if self._deviceHandle is None:
            return -1
        ret = 0
        open_flag = 0
        if streamFlag & 1: #color
            stream_Handle = streamHandle()
            if callback is None:
                if (self._hasColorDevice):
                    print("startStreams :colordevice Handle")
                    ret = berxelOpenStream(self._deviceColorHandle, 1, byref(stream_Handle))
                else:
                    print("startStreams :device Handle")
                    ret = berxelOpenStream(self._deviceHandle, 1, byref(stream_Handle))
            else:
                if (self._hasColorDevice):
                    ret = berxelOpenStream2(self._deviceColorHandle, 1, byref(stream_Handle), callback, user)
                else:
                    ret = berxelOpenStream2(self._deviceHandle, 1, byref(stream_Handle), callback, user)
            if(ret == 0):
                print("open color stream succeed")
                self._mColorStream = stream_Handle
                open_flag = (open_flag | 1)
            else:
                print("open color stream failed")

        if streamFlag & 2:#flag & BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM']:
            stream_Handle = streamHandle()
            print("open Depth Stream")
            if callback is None:
                ret = berxelOpenStream(self._deviceHandle, 2, byref(stream_Handle))
            else:
                ret = berxelOpenStream2(self._deviceHandle, 2, byref(stream_Handle), callback, user)
            print( "ret = " , ret)
            if ret == 0:
                print("open Depth Stream succeed")
                self._mDepthStream = stream_Handle
                open_flag = (open_flag | 2)
            else:
                print("open Depth Stream failed")

        if streamFlag & 4: #lag & BerxelHawkStreamType.forward_dict['BERXEL_HAWK_IR_STREAM']:
            stream_Handle = streamHandle()
            if callback is None:
                ret = berxelOpenStream(self._deviceHandle, 4, byref(stream_Handle))
            else:
                ret = berxelOpenStream2(self._deviceHandle, 4, byref(stream_Handle), callback, user)
            if (ret == 0):
                print("open Ir Stream succeed")
                self._mIrStream = stream_Handle
                open_flag = (open_flag | 4)
            else:
                print("open ir Stream failed")


        print("openFlga = ", open_flag, "streamFlag= ", streamFlag)

        if streamFlag == open_flag:
            print("start stream succeed")
            return 0
        else:
            print("start stream failed")
            return  -1

    def stopStream(self, streamFlag):
        if self._deviceHandle is None:
            return -1

        retColor = 0
        retDepth = 0
        retIr = 0

        if streamFlag & 1:
            if self._mColorStream is not  None:
                retColor = berxelCloseStream(self._mColorStream)
                if retColor ==0:
                    print("close color stream succeed")
                    self._mColorStream = None
                else:
                    print("close color stream failed")

        if streamFlag & 2:
            if(self._mDepthStream is not  None):
                retDepth = berxelCloseStream(self._mDepthStream)
                if retDepth ==0:
                    print("close depth stream succeed")
                    self._mDepthStream = None
                else:
                    print("close depth stream failed")

        if streamFlag & 4:
            if(self._mIrStream is not  None):
                retIr = berxelCloseStream(self._mIrStream)
                if retIr ==0:
                    print("close Ir stream succeed")
                    self._mIrStream = None
                else:
                    print("close Ir stream failed")

        if (retColor < 0) or (retDepth < 0) or  (retIr< 0):
            print("colose stream failed")
            return -1
        else:
            print("clsoe stream succeed1")
            return 0

    def releaseFrame(self, hawkFrame):
        if hawkFrame is None:
            return  -1
        else:
            berxelReleaseFrame(byref(hawkFrame.getFrameHandle()))

    def readColorFrame(self, timeout):
        if self._mColorStream is None:
            print("Color stream is not opened")
            return None
        frame_handle = imageFrameHandle()
        ret = berxelReadFrame(self._mColorStream, byref(frame_handle), timeout)
        if 0 == ret:
            return BerxelHawkFrame(frame_handle)
        else:
            # print("Read color stream failed")
            return None

    def readDepthFrame(self, timeout):
        if self._mDepthStream is None:
            print("Depth stream is not opened")
            return None

        # print("read depth frame")
        frame_handle = imageFrameHandle()
        ret = berxelReadFrame(self._mDepthStream, byref(frame_handle), timeout)
        if 0== ret:
            # print("pixel", frame_handle.contents.pixelType)
            # print("type ", frame_handle.contents.type)
            # print("frameIndex ", frame_handle.contents.frameIndex)
            # print("timestamp ", frame_handle.contents.timestamp)
            # print("fps ", frame_handle.contents.fps)
            # print("width ", frame_handle.contents.width)
            # print("height",frame_handle.contents.height)
            return BerxelHawkFrame(frame_handle)
        else:
            # print("Read depth stream failed")
            return None


    def readIrFrame(self, timeout):
        if self._mIrStream is None:
            print("Ir stream is not opened")
            return None

        frame_handle = imageFrameHandle()
        ret = berxelReadFrame(self._mIrStream, byref(frame_handle), timeout)

        if 0== ret:
            return BerxelHawkFrame(frame_handle)
        else:
            # print("Read ir stream failed")
            return None

    def getVersion(self):
        if self._deviceHandle is None:
            return None
        version_info = BerxelVersionInfo()
        ret = berxelGetVersion(self._deviceHandle, byref(version_info))
        if ret == 0:
            print( version_info.sdkVersion.major , version_info.sdkVersion.minor, version_info.sdkVersion.revision)
            return  version_info
        else:
            return None

    def getCurrentDeviceInfo(self):
        if self._deviceHandle is None:
            return None

        device_info =   BerxelHawkDeviceInfo()
        ret = berxelGetCurrentDeviceInfo(self._deviceHandle,byref(device_info))
        if ret == 0:
            print(device_info.serialNumber)
            return device_info
        else:
            return None

    def getDeviceIntrinsicParams(self):
        if self._deviceHandle is None:
            return None

        params  = BerxelHawkDeviceIntrinsicParams()
        ret = berxelGetDeviceIntrinsicParams(self._deviceHandle, byref(params))
        if  ret == 0:
            return params
        else:
            return None

    def setStreamMirror(self,bMirror):
        if self._deviceHandle is None:
            return -1

        ret = 0
        print("setStreamMirror =",bMirror)
        if bMirror == True:
            print("set Mirror Enable")
            if(self._hasColorDevice):
                ret = berxelSetStreamMirror(self._deviceColorHandle, 1)
            else:
                ret = berxelSetStreamMirror(self._deviceHandle,1)
        else:
            print("set Mirror disable")
            ret = berxelSetStreamMirror(self._deviceHandle,0)
        return ret

    def setRegistrationEnable(self, bEnable):
        if self._deviceHandle is None:
            return -1
        print("setRegistrationEnable =", bEnable)
        ret = 0
        if bEnable == True:
            ret = berxelEnableRegistration(self._deviceHandle,1)
        else:
            ret= berxelEnableRegistration(self._deviceHandle, 0)

        return ret

    def setFrameSync(self, bEnable):
        if self._deviceHandle is None:
            return -1
        ret = 0
        print("setFrameSync =", bEnable)

        if bEnable == True:
            ret = berxelSetFrameSync(self._deviceHandle,1)
        else:
            ret= berxelSetFrameSync(self._deviceHandle,0)

        return ret

    def setSystemClock(self):
        if self._deviceHandle is None:
            return -1

        print("set System Clock")
        ret = berxelSetSystemClock(self._deviceHandle)

        return ret

    def setDenoiseStatus(self, bEnable):
        if self._deviceHandle is None:
            return -1
        print("setDenoiseStatus =", bEnable)
        ret = 0
        if bEnable == True:
            ret = berxelSetDenoise(self._deviceHandle,1)
        else:
            ret = berxelSetDenoise(self._deviceHandle, 0)

        return ret
    
    def setColorQuality(self, nValue):
        if self._deviceHandle is None:
            return -1
        print("setColorQuality = ", nValue)

        ret = 0
        if(self._hasColorDevice):
            ret = berxelSetColorQuality(self._deviceColorHandle, nValue)
        else:
            ret = berxelSetColorQuality(self._deviceHandle, nValue)

        return ret

    def convertDepthToPoint(self, pData, width, height, factor, fx, fy, cx, cy , pixelType):
        if self._deviceHandle is None:
            return -1
        #print("setDenoiseStatus =",)

        ret = 0
        #tempPoint3DList = BerxelHawkPoint3D * 1280 *800
        #print("fx ,fy ,cx , cy, factor", fx, fy,cx, cy, factor)
        point3DList = BerxelHawkPoint3DList()

        ret = berxelConvertDepthToPointCloud(pData, width, height, factor, fx, fy ,cx, cy, byref(point3DList), pixelType)

        return point3DList

    def setColorExposureGain(self, exposureTime, gain):
        if self._deviceHandle is None:
            return -1

        ret = 0

        if (self._hasColorDevice):
            ret = berxelSetColorExposureGain(self._deviceColorHandle, exposureTime, gain)
        else:
            ret = berxelSetColorExposureGain(self._deviceHandle, exposureTime,  gain)
        return  ret

    def enableColorAutoExposure(self):
        if self._deviceHandle is None:
            return -1

        ret = 0
        if (self._hasColorDevice):
            ret = berxelRecoveryColorAE(self._deviceColorHandle)
        else:
            ret = berxelRecoveryColorAE(self._deviceHandle)
        return ret
    
    def setTemporalDenoiseStatus(self, bEnable):
        if self._deviceHandle is None:
            return -1
        print("setTemporalDenoiseStatus =", bEnable)
        ret = 0
        if bEnable == True:
            ret = berxelEnableTemporalDenoise(self._deviceHandle,1)
        else:
            ret = berxelEnableTemporalDenoise(self._deviceHandle, 0)

    def setSpatialDenoiseStatus(self, bEnable):
        if self._deviceHandle is None:
            return -1
        print("setSpatialDenoiseStatus =", bEnable)
        ret = 0
        if bEnable == True:
            ret = berxelEnableSpatialDenoise(self._deviceHandle,1)
        else:
            ret = berxelEnableSpatialDenoise(self._deviceHandle, 0)

    def setDepthElectricCurrent(self, nValue):
        if self._deviceHandle is None:
            return -1
        print("setDepthElectricCurrent = ", nValue)
        ret = berxelSetDepthElectricCurrent(self._deviceHandle, nValue)
        return ret
