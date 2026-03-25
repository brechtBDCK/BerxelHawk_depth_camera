#coding=utf-8

#from BerxelHawkNativeMethods import *
try:
    from .BerxelHawkDevice import *
except ImportError:
    from BerxelHawkDevice import *
import threading


class DeviceCallback(object):
    def __init__(self):
        self._user_callback = None
        self._real_callback = None

    def setCallback(self, userCallback):
        self._user_callback = userCallback
        if self._user_callback is None:
            self._real_callback = BerxelDeviceStatusCallback()
        else:
            self._real_callback = BerxelDeviceStatusCallback(self._deviceStateCallback)

    def _deviceStateCallback(self, deviceUri, serialNumber, deviceState, userData):
        self._user_callback(deviceUri, serialNumber, deviceState, userData)





class BerxelHawkContext(object):
    #_instance_lock = threading.Lock()
    mDeviceList = []
    mColorDeviceList = []
    mDeviceCount = c_uint32(0)
    mDeviceInfoHandle = deviceInfoHandle()
    deviceCallbackObj = DeviceCallback()

    target_color_devices = [
        (0x001F, 0x0603),
        (0x636F, 0x0C45),
        (0x001C, 0x0603)
    ]

    target_depth_devices = [
        (0x1012, 0x3558),
        (0x1002, 0x3558),
        (0x1013, 0x3558)
    ]

    def __init__(self):
        print("Test init")
        #berxelInit()

    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, '_instance'):
    #         print("new 1")
    #         with BerxelHawkContext._instance_lock:
    #             print("new 2")
    #             if not hasattr(cls, '_instance'):
    #                 print("new 3")
    #                 BerxelHawkContext._instance = super().__new__(cls)
    #                 berxelInit()
    #
    #         return BerxelHawkContext._instance

    # def __del__(self):
    #     print("release list")
    #     berxelReleaseDeviceList(byref(self.mDeviceInfoHandle))
    #     print("destroy2")
    #     berxelDestroy()


    # def destroy(self):
    #     print("destroy")
    #     berxelDestroy()

    def initCamera(self):
        print("init Camera")
        berxelInit()

    def destroyCamera(self):
        print("destroy Camera")
        berxelReleaseDeviceList(byref(self.mDeviceInfoHandle))
        berxelDestroy()

    def getDeviceList(self):
        # deviceInfo_Handle = deviceInfoHandle()
        self.mDeviceList = []
        self.mColorDeviceList = []
        berxelGetDeviceList(byref(self.mDeviceInfoHandle), byref(self.mDeviceCount))
        if self.mDeviceCount.value < 1:
            return self.mDeviceList
        else:
            for x in range(self.mDeviceCount.value):

                #if(self.mDeviceInfoHandle[x].vendorId == 0x0603 and self.mDeviceInfoHandle[x].productId == 0x001F) or (self.mDeviceInfoHandle[x].vendorId == 0x0C45 and self.mDeviceInfoHandle[x].productId == 0x636F ) or   (self.mDeviceInfoHandle[x].vendorId == 0x0603 and self.mDeviceInfoHandle[x].productId == 0x001C):

                if (self.mDeviceInfoHandle[x].productId , self.mDeviceInfoHandle[x].vendorId ) in self.target_color_devices:

                    self.mColorDeviceList.append(self.mDeviceInfoHandle[x])
                    print("color vid: ", self.mDeviceInfoHandle[x].vendorId)
                    print("color pid：", self.mDeviceInfoHandle[x].productId)
                    print("color addr：", self.mDeviceInfoHandle[x].deviceAddress)
                    print("color sn：", self.mDeviceInfoHandle[x].serialNumber)
                else:
                    self.mDeviceList.append(self.mDeviceInfoHandle[x])
                    print("vid: ", self.mDeviceInfoHandle[x].vendorId)
                    print("pid：", self.mDeviceInfoHandle[x].productId)
                    print("addr：", self.mDeviceInfoHandle[x].deviceAddress)
                    print("sn：", self.mDeviceInfoHandle[x].serialNumber)



            #berxelReleaseDeviceList(byref(deviceInfo_Handle))
        return self.mDeviceList


    def setDeviceStatusCallback(self, callback, data):
        BerxelHawkContext.deviceCallbackObj.setCallback(callback)
        return berxelSetDeviceStatusCallback(BerxelHawkContext.deviceCallbackObj._real_callback, data)

    def openDevice(self, deviceinfo):
        device_handle = deviceHandle()

        color_device_handle = deviceHandle()

        # if len(BerxelHawkContext.mDeviceList) < 1:
        #     print("not find device")
        #     return None
        #
        # print("openDevice-> addr：", BerxelHawkContext.mDeviceList[0].deviceAddress)


        ret1 = -1

        needOpendColorDevice = False

        if (deviceinfo.productId, deviceinfo.vendorId) in self.target_depth_devices:
            print("this device has color device ,need open color device too")
            needOpendColorDevice = True

            for x in range(0, len(self.mColorDeviceList)):
                print("depth SN  = ", deviceinfo.serialNumber, " color SN = ", self.mColorDeviceList[x].serialNumber)
                if(deviceinfo.serialNumber == self.mColorDeviceList[x].serialNumber):
                    ret1 = berxelOpenDeviceByAddr(self.mColorDeviceList[x].deviceAddress, byref(color_device_handle))
                    if(ret1 == 0):
                        print("open color device succeeded")
                    else:
                        print("open color device failed")






        if (needOpendColorDevice == True):

            if(ret1 == 0):
                print("re1 == 0 , start open depthDevice ")
                ret = berxelOpenDeviceByAddr(deviceinfo.deviceAddress, byref(device_handle))
                if (ret == 0):
                    print("ret == 0, open depth succeed")
                    hawkDevice = BerxelHawkDevice(device_handle)
                    hawkDevice.initColorHandle(color_device_handle)
                    return hawkDevice
            else:
                print("ret1 == -1 , return None")
                return None




        else:
            ret = berxelOpenDeviceByAddr(deviceinfo.deviceAddress, byref(device_handle))
            if (ret == 0):
                return BerxelHawkDevice(device_handle)
            else:
                return None




    def closeDevice(self, device):
        if device is None:
            return -1
        else:

            if(device._deviceColorHandle is not None):
                print("close color device hanle")
                berxelCloseDevice(device._deviceColorHandle)

            print("close  device hanle")
            return berxelCloseDevice(device._deviceHandle)
