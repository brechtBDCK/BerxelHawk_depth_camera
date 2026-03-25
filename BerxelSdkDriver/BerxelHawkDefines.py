#coding=utf-8
#!/usr/bin/python
#coding=utf-8

import os
import time
import datetime
import threading
import sys

import platform

current_system = platform.system()
if current_system == 'Windows':
    from ctypes import WINFUNCTYPE
else:
    from ctypes import CFUNCTYPE
#
# from ctypes import Structure, POINTER, \
#     c_uint8, c_uint16, c_uint32, c_uint64, \
#     c_int, c_int8, c_int16,c_int64, c_float, c_void_p, c_char_p, c_char

from ctypes import *


class Enum(object):
    def __init__(self, member_dict, scope_dict=None):
        if scope_dict is None:
            # Affect caller's locals, not this module's.
            scope_dict = sys._getframe(1).f_locals
        forward_dict = {}
        reverse_dict = {}
        next_value = 0
        for name, value in member_dict.items():
            if value is None:
                value = next_value
                next_value += 1
            forward_dict[name] = value
            if value in reverse_dict:
                raise ValueError('Multiple names for value %r: %r, %r' % (
                    value, reverse_dict[value], name
                ))
            reverse_dict[value] = name
            scope_dict[name] = value
        self.forward_dict = forward_dict
        self.reverse_dict = reverse_dict

    def __call__(self, value):
        return self.reverse_dict[value]

    def get(self, value, default=None):
        return self.reverse_dict.get(value, default)

BerxelHawkPixelType = Enum({
        'BERXEL_HAWK_PIXEL_TYPE_IMAGE_RGB24': 0x00,
        'BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_12I_4D': 0x01,
        'BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_13I_3D': 0x02,
        'BERXEL_HAWK_PIXEL_TYPE_IR_16BIT': 0x03,
        'BERXEL_HAWK_PIXEL_INVALID_TYPE': 0xff,
})


BerxelHawkStreamType = Enum({
        'BERXEL_HAWK_COLOR_STREAM': 0x01,
        'BERXEL_HAWK_DEPTH_STREAM': 0x02,
        'BERXEL_HAWK_IR_STREAM': 0x04,
        'BERXEL_HAWK_PIXEL_FORMAT_MIX_16BIT': 0x20,
        'BERXEL_HAWK_LIGHT_IR_STREAM': 0xff,
})


BerxelHawkStreamFlagMode = Enum({
        'BERXEL_HAWK_SINGULAR_STREAM_FLAG_MODE': 0x01,
        'BERXEL_HAWK_MIX_STREAM_FLAG_MODE': 0x02,
        'BERXEL_HAWK_MIX_HD_STREAM_FLAG_MODE': 0x03,
})

BerxelHawkDeviceStatus = Enum({
        'BERXEL_HAWK_DEVICE_CONNECT': 0x00,
        'BERXEL_HAWK_DEVICE_DISCONNECT': 0x01,
})


class BerxelDevice(Structure):
    pass

deviceHandle = POINTER(BerxelDevice)
deviceHandle_p = POINTER(deviceHandle)



class BerxelStream(Structure):
    pass


streamHandle = POINTER(BerxelStream)
streamHandle_p = POINTER(streamHandle)


class BerxelHawkSdkVersion(Structure):
    _pack_ = 1
    _fields_ = [
        ('major', c_uint16),
        ('minor', c_uint16),
        ('revision', c_uint16),
    ]



class BerxelHawkFwVersion(Structure):
    _pack_ = 1
    _fields_ = [
        ('major', c_uint16),
        ('minor', c_uint16),
        ('revision', c_uint16),
        ('chipVersion', c_char * 64)
    ]

class BerxelHawkHwVersion(Structure):
    _pack_ = 1
    _fields_ = [
        ('major', c_uint16),
        ('minor', c_uint16),
        ('revision', c_uint16)
    ]


class BerxelVersionInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ('sdkVersion', BerxelHawkSdkVersion),
        ('fwVersion', BerxelHawkFwVersion),
        ('hwVersion', BerxelHawkHwVersion)
    ]

versionHandle = POINTER(BerxelVersionInfo)




class BerxelHawkImageFrame(Structure):
    _pack_ = 1
    _fields_ = [
        ('pixelType', c_uint32),  # ImiPixelFormat in fact:Structure fileds can not be non-C type
        ('type', c_uint32),
        ('frameIndex', c_uint32),
        ('timestamp', c_uint64),
        ('fps', c_uint32),
        ('width', c_uint32),
        ('height', c_uint32),
        ('dataSize', c_uint32),
        ('pVoidData', c_void_p),

        ]


imageFrameHandle = POINTER(BerxelHawkImageFrame)
imageFrameHandle_p = POINTER(imageFrameHandle)


# device attribute
class BerxelHawkDeviceInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ('vendorId', c_uint16),
        ('productId', c_uint16),
        ('deviceNum', c_uint32),
        ('deviceType', c_uint32),
        ('deviceBus', c_uint32),
        ('devicePort', c_char * 32),
        ('serialNumber', c_char * 32),
        ('deviceAddress', c_char * 255),
        ('bcdDevice', c_uint16),
        ('location', c_char * 256)]


deviceInfoHandle = POINTER(BerxelHawkDeviceInfo)
deviceInfoHandle_p = POINTER(deviceInfoHandle)


class BerxelHawkCameraIntrinsic(Structure):
    _pack_ = 1
    _fields_ = [
        ('fx', c_float),
        ('fy', c_float),
        ('cx', c_float),
        ('cy', c_float),
        ('k1', c_float),
        ('k2', c_float),
        ('p1', c_float),
        ('p2', c_float),
        ('k3', c_float)]


class BerxelHawkCameraRotateIntrinsic(Structure):
    _pack_ = 1
    _fields_ = [
        ('rotate1', c_float),
        ('rotate2', c_float),
        ('rotate3', c_float),
        ('rotate4', c_float),
        ('rotate5', c_float),
        ('rotate6', c_float),
        ('rotate7', c_float),
        ('rotate8', c_float),
        ('rotate9', c_float)]


class BerxelHawkCameraTranslationIntrinsic(Structure):
    _pack_ = 1
    _fields_ = [
        ('translation1', c_float),
        ('translation2', c_float),
        ('translation3', c_float)]



CameraIntrinsicParamsHandle = POINTER(BerxelHawkCameraIntrinsic)
CameraIntrinsicParamsHandle_p = POINTER(CameraIntrinsicParamsHandle)


class BerxelHawkDeviceIntrinsicParams(Structure):
    _pack_ = 1
    _fields_ = [
        ('colorIntrinsicParams',  BerxelHawkCameraIntrinsic),
        ('irIntrinsicParams', BerxelHawkCameraIntrinsic),
        ('liteIrIntrinsicParams', BerxelHawkCameraIntrinsic),
        ('rotateIntrinsicParams', BerxelHawkCameraRotateIntrinsic),
        ('translationIntrinsicParams', BerxelHawkCameraTranslationIntrinsic)]

deviceIntrinsicParamsHandle = POINTER(BerxelHawkDeviceIntrinsicParams)
deviceIntrinsicParamsHandle_p = POINTER(deviceIntrinsicParamsHandle)

class BerxelHawkStreamFrameMode(Structure):
    _pack_ = 1
    _fields_ = [
        ('pixelFormat', c_uint32),
        ('resolutionX', c_int16),
        ('resolutionY', c_int16),
        ('framerate', c_int8)]


frameModeHandle = POINTER(BerxelHawkStreamFrameMode)
frameModeHandle_p = POINTER(frameModeHandle)



class BerxelHawkPoint3D(Structure):
    _pack_ = 1
    _fields_ = [
    ('x', c_float),
    ('y', c_float),
    ('z', c_float)]

class BerxelHawkPoint3DList(Structure):
    _fields_ = [
    ('point3DList', BerxelHawkPoint3D * 1024000)]

pointListHandle = POINTER(BerxelHawkPoint3DList)

if current_system == 'Windows':
    BerxelDeviceStatusCallback = WINFUNCTYPE(None, c_char_p, c_char_p, c_int, c_void_p)
    BerxelNewFrameCallback = WINFUNCTYPE(None, streamHandle, imageFrameHandle, c_void_p)
else:
    BerxelDeviceStatusCallback = CFUNCTYPE(None, c_char_p, c_char_p, c_int, c_void_p)
    BerxelNewFrameCallback = CFUNCTYPE(None, streamHandle, imageFrameHandle, c_void_p)

