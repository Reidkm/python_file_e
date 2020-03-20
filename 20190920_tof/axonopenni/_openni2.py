#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Auto-generated file; do not edit directly
# Tue Nov 12 13:59:13 2013
import sys

import ctypes
from axonopenni.utils import CEnum, UnloadedDLL

TRUE = 1
FALSE = 0
ONI_MAX_STR = 256
ONI_MAX_SENSORS = 10
ONI_VERSION_MAJOR = 2
ONI_VERSION_MINOR = 2
ONI_VERSION_MAINTENANCE = 0
ONI_VERSION_BUILD = 30
ONI_VERSION = (ONI_VERSION_MAJOR * 100000000 + ONI_VERSION_MINOR *
               1000000 + ONI_VERSION_MAINTENANCE * 10000 + ONI_VERSION_BUILD)


def ONI_CREATE_API_VERSION(major, minor): return ((major) * 1000 + (minor))


ONI_API_VERSION = ONI_CREATE_API_VERSION(ONI_VERSION_MAJOR, ONI_VERSION_MINOR)
XN_DEVICE_MAX_STRING_LENGTH = 200
XN_IO_MAX_I2C_BUFFER_SIZE = 10
XN_MAX_LOG_SIZE = (6 * 1024)
XN_MAX_VERSION_MODIFIER_LENGTH = 16


def _get_calling_conv(*args):
    if sys.platform == 'win32':
        return ctypes.WINFUNCTYPE(*args)
    else:
        return ctypes.CFUNCTYPE(*args)


class OniStatus(CEnum):
    _names_ = {'ONI_STATUS_NOT_SUPPORTED': 3,
               'ONI_STATUS_OK': 0,
               'ONI_STATUS_TIME_OUT': 102,
               'ONI_STATUS_ERROR': 1,
               'ONI_STATUS_NOT_IMPLEMENTED': 2,
               'ONI_STATUS_NO_DEVICE': 6,
               'ONI_STATUS_OUT_OF_FLOW': 5,
               'ONI_STATUS_BAD_PARAMETER': 4}
    _values_ = {0: 'ONI_STATUS_OK',
                1: 'ONI_STATUS_ERROR',
                2: 'ONI_STATUS_NOT_IMPLEMENTED',
                3: 'ONI_STATUS_NOT_SUPPORTED',
                4: 'ONI_STATUS_BAD_PARAMETER',
                5: 'ONI_STATUS_OUT_OF_FLOW',
                6: 'ONI_STATUS_NO_DEVICE',
                102: 'ONI_STATUS_TIME_OUT'}
    ONI_STATUS_OK = 0
    ONI_STATUS_ERROR = 1
    ONI_STATUS_NOT_IMPLEMENTED = 2
    ONI_STATUS_NOT_SUPPORTED = 3
    ONI_STATUS_BAD_PARAMETER = 4
    ONI_STATUS_OUT_OF_FLOW = 5
    ONI_STATUS_NO_DEVICE = 6
    ONI_STATUS_TIME_OUT = 102


class OniSensorType(CEnum):
    _names_ = {'ONI_SENSOR_IR': 1,
               'ONI_SENSOR_COLOR': 2,
               'ONI_SENSOR_DEPTH': 3}
    _values_ = {1: 'ONI_SENSOR_IR',
                2: 'ONI_SENSOR_COLOR',
                3: 'ONI_SENSOR_DEPTH'}
    ONI_SENSOR_IR = 1
    ONI_SENSOR_COLOR = 2
    ONI_SENSOR_DEPTH = 3


class OniPixelFormat(CEnum):
    _names_ = {'ONI_PIXEL_FORMAT_RGB888': 200,
               'ONI_PIXEL_FORMAT_JPEG': 204,
               'ONI_PIXEL_FORMAT_GRAY8': 202,
               'ONI_PIXEL_FORMAT_DEPTH_1_MM': 100,
               'ONI_PIXEL_FORMAT_YUYV': 205,
               'ONI_PIXEL_FORMAT_YUV422': 201,
               'ONI_PIXEL_FORMAT_SHIFT_9_3': 103,
               'ONI_PIXEL_FORMAT_SHIFT_9_2': 102,
               'ONI_PIXEL_FORMAT_GRAY16': 203,
               'ONI_PIXEL_FORMAT_DEPTH_100_UM': 101,
               'ONI_PIXEL_FORMAT_DEPTH_1_2_MM':111,
               'ONI_PIXEL_FORMAT_DEPTH_1_3_MM':110}
    _values_ = {100: 'ONI_PIXEL_FORMAT_DEPTH_1_MM',
                101: 'ONI_PIXEL_FORMAT_DEPTH_100_UM',
                102: 'ONI_PIXEL_FORMAT_SHIFT_9_2',
                103: 'ONI_PIXEL_FORMAT_SHIFT_9_3',
                110: 'ONI_PIXEL_FORMAT_DEPTH_1_3_MM',
                111: 'ONI_PIXEL_FORMAT_DEPTH_1_2_MM',
                200: 'ONI_PIXEL_FORMAT_RGB888',
                201: 'ONI_PIXEL_FORMAT_YUV422',
                202: 'ONI_PIXEL_FORMAT_GRAY8',
                203: 'ONI_PIXEL_FORMAT_GRAY16',
                204: 'ONI_PIXEL_FORMAT_JPEG',
                205: 'ONI_PIXEL_FORMAT_YUYV'}
    ONI_PIXEL_FORMAT_DEPTH_1_MM = 100
    ONI_PIXEL_FORMAT_DEPTH_100_UM = 101
    ONI_PIXEL_FORMAT_DEPTH_1_3_MM = 110
    ONI_PIXEL_FORMAT_DEPTH_1_2_MM = 111
    ONI_PIXEL_FORMAT_SHIFT_9_2 = 102
    ONI_PIXEL_FORMAT_SHIFT_9_3 = 103
    ONI_PIXEL_FORMAT_RGB888 = 200
    ONI_PIXEL_FORMAT_YUV422 = 201
    ONI_PIXEL_FORMAT_GRAY8 = 202
    ONI_PIXEL_FORMAT_GRAY16 = 203
    ONI_PIXEL_FORMAT_JPEG = 204
    ONI_PIXEL_FORMAT_YUYV = 205


class OniDeviceState(CEnum):
    _names_ = {'ONI_DEVICE_STATE_ERROR': 1,
               'ONI_DEVICE_STATE_NOT_READY': 2,
               'ONI_DEVICE_STATE_EOF': 3,
               'ONI_DEVICE_STATE_OK': 0}
    _values_ = {0: 'ONI_DEVICE_STATE_OK',
                1: 'ONI_DEVICE_STATE_ERROR',
                2: 'ONI_DEVICE_STATE_NOT_READY',
                3: 'ONI_DEVICE_STATE_EOF'}
    ONI_DEVICE_STATE_OK = 0
    ONI_DEVICE_STATE_ERROR = 1
    ONI_DEVICE_STATE_NOT_READY = 2
    ONI_DEVICE_STATE_EOF = 3


class OniImageRegistrationMode(CEnum):
    _names_ = {'ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR': 1,
               'ONI_IMAGE_REGISTRATION_DEPTH_IR_TO_COLOR': 2,
               'ONI_IMAGE_REGISTRATION_COLOR_TO_DEPTH': 3,
               'ONI_IMAGE_REGISTRATION_COLOR_UNDISTORTION_ONLY': 4,
               'ONI_IMAGE_REGISTRATION_OFF': 0}
    _values_ = {0: 'ONI_IMAGE_REGISTRATION_OFF',
                2: 'ONI_IMAGE_REGISTRATION_DEPTH_IR_TO_COLOR',
                3: 'ONI_IMAGE_REGISTRATION_COLOR_TO_DEPTH',
                4: 'ONI_IMAGE_REGISTRATION_COLOR_UNDISTORTION_ONLY',
                1: 'ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR'}
    ONI_IMAGE_REGISTRATION_OFF = 0
    ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR = 1
    ONI_IMAGE_REGISTRATION_DEPTH_IR_TO_COLOR = 2
    ONI_IMAGE_REGISTRATION_COLOR_TO_DEPTH = 3
    ONI_IMAGE_REGISTRATION_COLOR_UNDISTORTION_ONLY = 4
# ONI_IMAGE_REGISTRATION_OFF = OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_OFF
# ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR = OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR
# ONI_IMAGE_REGISTRATION_DEPTH_IR_TO_COLOR = OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_IR_TO_COLOR
# ONI_IMAGE_REGISTRATION_COLOR_TO_DEPTH = OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_COLOR_TO_DEPTH
# ONI_IMAGE_REGISTRATION_COLOR_UNDISTORTION_ONLY = OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_COLOR_UNDISTORTION_ONLY

class _anon_enum_5(CEnum):
    _names_ = {'ONI_TIMEOUT_NONE': 0,
               'ONI_TIMEOUT_FOREVER': -1}
    _values_ = {0: 'ONI_TIMEOUT_NONE',
                -1: 'ONI_TIMEOUT_FOREVER'}
    ONI_TIMEOUT_NONE = 0
    ONI_TIMEOUT_FOREVER = -1


ONI_TIMEOUT_NONE = _anon_enum_5.ONI_TIMEOUT_NONE
ONI_TIMEOUT_FOREVER = _anon_enum_5.ONI_TIMEOUT_FOREVER


class OniCallbackHandleImpl(ctypes.Structure):

    def __repr__(self):
        return 'OniCallbackHandleImpl()' % ()

### AXONLink
class AXonLinkSerialNumber(ctypes.Structure):
    serial = 'ctypes.c_byte'
    def __repr__(self):
        return (''.join([chr(s) for s in self.serial]))


class CamIntrinsicParam(ctypes.Structure): #AXON内参
    ResolutionX = 'ctypes.c_int'
    ResolutionY = 'ctypes.c_int'
    fx = 'ctypes.c_float'
    fy = 'ctypes.c_float'
    cx = 'ctypes.c_float'
    cy = 'ctypes.c_float'
    k1 = 'ctypes.c_float'
    k2 = 'ctypes.c_float'
    k3 = 'ctypes.c_float'
    p1 = 'ctypes.c_float'
    p2 = 'ctypes.c_float'
    k4 = 'ctypes.c_float'
    k5 = 'ctypes.c_float'
    k6 = 'ctypes.c_float'
    def __repr__(self):
        return "[ResolutionX=%r,ResolutionY=%r,fx=%r,fy=%r,cx=%r,cy=%r,k1=%r,k2=%r,k3=%r,p1=%r,p2=%r,k4=%r,k5=%r,k6=%r]\n" % \
               (self.ResolutionX, self.ResolutionY,self.fx,self.fy,self.cx,self.cy,self.k1,self.k2,self.k3,self.p1,self.p2,self.k4,self.k5,self.k6)


class CamExtrinsicParam(ctypes.Structure):  # AXON相机外参
    R_Param = 'ctypes.c_float'
    T_Param = 'ctypes.c_float'
    def __repr__(self):
        return "R_Param:%r,\nT_Param:%r"%([i for i in self.R_Param], [j for j in self.T_Param])

class AXonLinkCamParam(ctypes.Structure):  # AXON相机参数
    stExtParam = 'CamExtrinsicParam'
    astDepthParam = 'CamIntrinsicParam'
    astColorParam = 'CamIntrinsicParam'

    def get_data(self):
        ext_param = self.stExtParam
        depth_param = self.astDepthParam
        color_param = self.astColorParam
        param = [ext_param, depth_param, color_param]
        return param

    def __repr__(self):
        return "**AXonLinkCamParam Values**\n*CamExtrinsicParam Value:\n%r\n\nDepthIntrinsicParam Value:\n%r\nColorIntrinsicParam Value:\n%r" % \
               (self.stExtParam, [d for d in self.astDepthParam], [c for c in self.astColorParam])


class AXonLinkFWVersion(ctypes.Structure):
    m_nhwType = 'ctypes.c_uint'
    m_nMajor = 'ctypes.c_uint'
    m_nMinor = 'ctypes.c_uint'
    m_nmaintenance = 'ctypes.c_uint'
    m_nbuild = 'ctypes.c_uint'
    m_nyear = 'ctypes.c_ubyte'
    m_nmonth = 'ctypes.c_ubyte'
    m_nday = 'ctypes.c_ubyte'
    m_nhour = 'ctypes.c_ubyte'
    m_nmin = 'ctypes.c_ubyte'
    m_nsec = 'ctypes.c_ubyte'
    reserved = 'ctypes.c_ubyte'
    def __repr__(self):
        self.SW = (self.m_nMajor, self.m_nMinor, self.m_nmaintenance, self.m_nbuild)
        return '(HW=%r, SW=%r.%r.%r.%r, BuildTime = %r/%r/%r-%r:%r:%r)' % \
               (self.m_nhwType, self.m_nMajor, self.m_nMinor, self.m_nmaintenance, self.m_nbuild, self.m_nyear, self.m_nmonth, self.m_nday,self.m_nhour,self.m_nmin,self.m_nsec)

class AXonLinkSWVersion(ctypes.Structure):
    major = 'ctypes.c_uint'
    minor = 'ctypes.c_uint'
    maintenance = 'ctypes.c_uint'
    build = 'ctypes.c_uint'

    def __repr__(self):
        return '%r.%r.%r.%r' % (self.major, self.minor, self.maintenance, self.build)


class AxonLinkFirmWarePacketVersion(ctypes.Structure):
    filename = 'ctype.c_char'
    hwType = 'ctype.c_uint'
    swVersion = 'AXonLinkSWVersion'
    def __repr__(self):
        return 'AxonLinkFirmWarePacketVersion(filename = %r, hwType = %r, swVersion = %r)' % (str(self.filename, encoding='utf8'), ord(self.hwType), self.swVersion)

class I2cValue(ctypes.Structure):
    pass

class E2Reg(ctypes.Structure):
    tpye = 'ctype.c_short'
    length = 'ctype.c_short'
    crc = 'ctype.c_short'
    data = 'ctype.c_char_p'
    def __repr__(self):
        return 'E2Reg(tpye = %r, length = %r, crc = %r,data = %r)' % \
               (self.tpye, self.length, self.crc, self.data)


class AXonDSPInterface(ctypes.Structure):
    UNDISTORT = 'ctype.c_uint'
    MASK = 'ctype.c_uint'
    NR3 = 'ctype.c_uint'
    NR2 = 'ctype.c_uint'
    GAMMA = 'ctype.c_uint'
    FLYING = 'ctype.c_uint'
    FLYING_2 = 'ctype.c_uint'
    R2Z = 'ctype.c_uint'
    reserved = 'ctype.c_uint'
    def __repr__(self):
        return 'AXonDSPInterface(UNDISTORT = %r, MASK = %r, NR3 = %r,NR2 = %r, GAMMA = %r, FLYING = %r,FLYING_2 = %r, R2Z = %r, reserved = %r)' % \
               (self.UNDISTORT, self.MASK, self.NR3, self.NR2, self.GAMMA, self.FLYING, self.FLYING_2, self.R2Z, self.reserved)


class AXonLinkSetExposureLevel(ctypes.Structure):
    curLevel = 'ctype.c_ubyte'
    write2E2flag = 'ctype.c_ubyte'
    reserved = 'ctype.c_ubyte'
    def __repr__(self):
        return 'AXonLinkSetExposureLevel(curLevel = %r, write2E2flag = %r, reserved = %r)' % \
               (self.curLevel, self.write2E2flag, self.reserved)


class AXonLinkGetExposureLevel(ctypes.Structure):
    custonID = 'ctype.c_ubyte'
    maxLevel = 'ctype.c_ubyte'
    curLevel = 'ctype.c_ubyte'
    reserved = 'ctype.c_ubyte'

    def __repr__(self):
        return 'AXonLinkGetExposureLevel(custonID = %r, maxLevel = %r, curLevel = %r, reserved = %r)' % (
        self.custonID, self.maxLevel, self.custonID, self.reserved)


class AXonCropping(ctypes.Structure):
    originX = 'ctype.c_ushort'
    originY = 'ctype.c_ushort'
    width = 'ctype.c_ushort'
    height = 'ctype.c_ushort'
    gx = 'ctype.c_ushort'
    gy = 'ctype.c_ushort'
    def __repr__(self):
        return 'AXonCropping(originX = %r, originY = %r, width = %r, height = %r, gx = %r, gy = %r)' % (
        self.originX, self.originY, self.width, self.height, self.gx, self.gy)

class AXonLinkReadE2OnType(ctypes.Structure):
    type = 'ctype.c_ushort'
    Length = 'ctype.c_ushort'
    data = 'ctype.c_ubyte'
    def __repr__(self):
        return 'AXonLinkReadE2OnType(type = %r, Length = %r, data = %r)' % (
        self.type, self.Length, self.data)

class AXonCalibration(ctypes.Structure):
    enable = 'ctype.c_ubyte'
    reserved = 'ctype.c_ubyte'
    def __repr__(self):
        return 'AXonCalibration(enable = %r, reserved = %r)' % (
        self.enable, self.reserved)

class AXonMotionThreshold(ctypes.Structure):
    thresHold = 'ctype.c_ushort'
    count = 'ctype.c_uint'
    remain = 'ctype.c_ushort'
    def __repr__(self):
        return 'AXonMotionThreshold(thresHold = %r, count = %r, remain = %r)' % (
        self.thresHold, self.count, self.remain)

class AXonBoard_SN(ctypes.Structure):
    len = 'ctype.c_ushort'
    serialNumber = 'ctype.c_char'
    def __repr__(self):
        return 'AXonBoard_SN(len = %r, serialNumber = %r)' % (
        self.len, self.serialNumber)

class AXonSensorExposureWindow(ctypes.Structure):
    cam_id = 'ctype.c_ubyte'
    auto_mode = 'ctype.c_ubyte'
    x = 'ctype.c_ushort'
    y = 'ctype.c_ushort'
    dx = 'ctype.c_ushort'
    dy = 'ctype.c_ushort'
    def __repr__(self):
        return 'AXonSensorExposureWindow(cam_id = %r, auto_mode = %r, x = %r, y = %r, dx = %r, dy = %r)' % (
            self.cam_id, self.auto_mode, self.x, self.y, self.dx, self.dy)


class OniVersion(ctypes.Structure):
    major = 'ctypes.c_int'
    minor = 'ctypes.c_int'
    maintenance = 'ctypes.c_int'
    build = 'ctypes.c_int'

    def __repr__(self):
        return '(major = %r, minor = %r, maintenance = %r, build = %r)' % (self.major, self.minor, self.maintenance, self.build)


class OniVideoMode(ctypes.Structure):
    pixelFormat = 'OniPixelFormat'
    resolutionX = 'ctypes.c_int'
    resolutionY = 'ctypes.c_int'
    fps = 'ctypes.c_int'

    def __repr__(self):
        return 'OniVideoMode(pixelFormat = %r, resolutionX = %r, resolutionY = %r, fps = %r)' % (self.pixelFormat, self.resolutionX, self.resolutionY, self.fps)


class OniSensorInfo(ctypes.Structure):
    sensorType = 'OniSensorType'
    numSupportedVideoModes = 'ctypes.c_int'
    pSupportedVideoModes = 'ctypes.POINTER(OniVideoMode)'

    def __repr__(self):
        return 'OniSensorInfo(sensorType = %r, numSupportedVideoModes = %r, pSupportedVideoModes = %r)' % (self.sensorType, self.numSupportedVideoModes, self.pSupportedVideoModes)


class OniDeviceInfo(ctypes.Structure):
    uri = '(ctypes.c_char * 256)'
    vendor = '(ctypes.c_char * 256)'
    name = '(ctypes.c_char * 256)'
    usbVendorId = 'ctypes.c_ushort'
    usbProductId = 'ctypes.c_ushort'

    def __repr__(self):
        return 'OniDeviceInfo(uri = %r, vendor = %r, name = %r, usbVendorId = %r, usbProductId = %r)' % (self.uri, self.vendor, self.name, self.usbVendorId, self.usbProductId)


class _OniDevice(ctypes.Structure):

    def __repr__(self):
        return '_OniDevice()' % ()


class _OniStream(ctypes.Structure):

    def __repr__(self):
        return '_OniStream()' % ()


class _OniRecorder(ctypes.Structure):

    def __repr__(self):
        return '_OniRecorder()' % ()


class OniFrame(ctypes.Structure):
    dataSize = 'ctypes.c_int'
    data = 'ctypes.c_void_p'
    sensorType = 'OniSensorType'
    timestamp = 'ctypes.c_ulonglong'
    frameIndex = 'ctypes.c_int'
    width = 'ctypes.c_int'
    height = 'ctypes.c_int'
    videoMode = 'OniVideoMode'
    croppingEnabled = 'OniBool'
    cropOriginX = 'ctypes.c_int'
    cropOriginY = 'ctypes.c_int'
    stride = 'ctypes.c_int'

    def __repr__(self):
        return 'OniFrame(dataSize = %r, data = %r, sensorType = %r, timestamp = %r, frameIndex = %r, width = %r, height = %r, videoMode = %r, croppingEnabled = %r, cropOriginX = %r, cropOriginY = %r, stride = %r)' % (self.dataSize, self.data, self.sensorType, self.timestamp, self.frameIndex, self.width, self.height, self.videoMode, self.croppingEnabled, self.cropOriginX, self.cropOriginY, self.stride)


class OniDeviceCallbacks(ctypes.Structure):
    deviceConnected = 'OniDeviceInfoCallback'
    deviceDisconnected = 'OniDeviceInfoCallback'
    deviceStateChanged = 'OniDeviceStateCallback'

    def __repr__(self):
        return 'OniDeviceCallbacks(deviceConnected = %r, deviceDisconnected = %r, deviceStateChanged = %r)' % (self.deviceConnected, self.deviceDisconnected, self.deviceStateChanged)


class OniCropping(ctypes.Structure):
    enabled = 'ctypes.c_int'
    originX = 'ctypes.c_int'
    originY = 'ctypes.c_int'
    width = 'ctypes.c_int'
    height = 'ctypes.c_int'

    def __repr__(self):
        return 'OniCropping(enabled = %r, originX = %r, originY = %r, width = %r, height = %r)' % (self.enabled, self.originX, self.originY, self.width, self.height)


class OniRGB888Pixel(ctypes.Structure):
    _packed_ = 1
    r = 'ctypes.c_ubyte'
    g = 'ctypes.c_ubyte'
    b = 'ctypes.c_ubyte'

    def __repr__(self):
        return 'OniRGB888Pixel(r = %r, g = %r, b = %r)' % (self.r, self.g, self.b)


class OniYUV422DoublePixel(ctypes.Structure):
    _packed_ = 1
    u = 'ctypes.c_ubyte'
    y1 = 'ctypes.c_ubyte'
    v = 'ctypes.c_ubyte'
    y2 = 'ctypes.c_ubyte'

    def __repr__(self):
        return 'OniYUV422DoublePixel(u = %r, y1 = %r, v = %r, y2 = %r)' % (self.u, self.y1, self.v, self.y2)


class OniSeek(ctypes.Structure):
    frameIndex = 'ctypes.c_int'
    stream = 'OniStreamHandle'

    def __repr__(self):
        return 'OniSeek(frameIndex = %r, stream = %r)' % (self.frameIndex, self.stream)


### AXONLink enum
class AXONLINK(CEnum):
    _names_ = {
                'AXONLINK_DEVICE_PROPERTY_GET_SOFTWARE_VERSION': 1610612737,
                'AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS': 1610612752,
                'AXONLINK_DEVICE_INVOKE_GET_E2PROM_ITEM': 1610613104,
                'AXONLINK_DEVICE_INVOKE_SET_REBOOT': 1610612864,
                'AXONLINK_DEVICE_INVOKE_GET_FWVERSION': 1610613088,
                'AXONLINK_DEVICE_INVOKE_GET_UPGRADE_STATUS': 1610612768,
                'AXONLINK_DEVICE_INVOKE_SET_UPLOADFILE': 1610612784,
                'AXONLINK_DEVICE_INVOKE_SET_UPGRADE_ENABLE': 1610612800,

                'AXONLINK_DEVICE_INVOKE_GET_REGE2_STATUS':1610612880,
                'AXONLINK_DEVICE_INVOKE_SET_REGE2FILE':1610612992,
                'AXONLINK_DEVICE_INVOKE_SET_REGE2ENABLE':1610613008,
                'AXONLINK_DEVICE_INVOKE_SET_DSP_DPINTERFACE':1610613024,
                'AXONLINK_DEVICE_INVOKE_GET_DSP_DPINTERFACE':1610613040,
                'AXONLINK_DEVICE_INVOKE_SET_CAMERA_TRIGER_SYNCSIGNAL':1610613120,
                'AXONLINK_DEVICE_INVOKE_SET_EXPOSURE_WINDOW':1610613136,

                'AXONLINK_DEVICE_INVOKE_EXTEND_BASE':1342177280,
                'AXONLINK_DEVICE_INVOKE_EXTEND_MANUALEXPOSURE':1342177281,
                'AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_SHUTTER':1342177282,
                'AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_SHUTTER':1342177283,
                'AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_GAIN':1342177284,
                'AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_GAIN':1342177285,

                'AXONLINK_DEVICE_COLOR_SENSOR_I2C':1610612816,
                'AXONLINK_DEVICE_DEPTH_SENSOR_I2C':1610612832,
                'AXONLINK_DEVICE_E2PROM':1610612848,

                'AXONLINK_STREAM_PROPERTY_FLIP':1627389953,
                'AXONLINK_STREAM_PROPERTY_CROPPING':1644167169,
                'AXONLINK_STREAM_PROPERTY_CALIBRATION':1660944385,
                'AXONLINK_STREAM_PROPERTY_MOTIONTHRESHOLD':1677721601,
                'AXONLINK_STREAM_PROPERTY_EXPOSURE_LEVEL':1694498817

                }
    _values_ = {
                1610612737: 'AXONLINK_DEVICE_PROPERTY_GET_SOFTWARE_VERSION',
                1610612752: 'AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS',
                1610612864: 'AXONLINK_DEVICE_INVOKE_SET_REBOOT',
                1610613088: 'AXONLINK_DEVICE_INVOKE_GET_FWVERSION',
                1610612768: 'AXONLINK_DEVICE_INVOKE_GET_UPGRADE_STATUS',
                1610612784: 'AXONLINK_DEVICE_INVOKE_SET_UPLOADFILE',
                1610612800: 'AXONLINK_DEVICE_INVOKE_SET_UPGRADE_ENABLE',
                1610613104: 'AXONLINK_DEVICE_INVOKE_GET_E2PROM_ITEM',
                1610612880: 'AXONLINK_DEVICE_INVOKE_GET_REGE2_STATUS',
                1610612992: 'AXONLINK_DEVICE_INVOKE_SET_REGE2FILE',
                1610613008: 'AXONLINK_DEVICE_INVOKE_SET_REGE2ENABLE',
                1610613024: 'AXONLINK_DEVICE_INVOKE_SET_DSP_DPINTERFACE',
                1610613040: 'AXONLINK_DEVICE_INVOKE_GET_DSP_DPINTERFACE',
                1610613120: 'AXONLINK_DEVICE_INVOKE_SET_CAMERA_TRIGER_SYNCSIGNAL',
                1610613136: 'AXONLINK_DEVICE_INVOKE_SET_EXPOSURE_WINDOW',

                1342177280: 'AXONLINK_DEVICE_INVOKE_EXTEND_BASE',
                1342177281: 'AXONLINK_DEVICE_INVOKE_EXTEND_MANUALEXPOSURE',
                1342177282: 'AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_SHUTTER',
                1342177283: 'AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_SHUTTER',
                1342177284: 'AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_GAIN',
                1342177285: 'AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_GAIN',

                1610612816: 'AXONLINK_DEVICE_COLOR_SENSOR_I2C',
                1610612832: 'AXONLINK_DEVICE_DEPTH_SENSOR_I2C',
                1610612848: 'AXONLINK_DEVICE_E2PROM',

                1627389953: 'AXONLINK_STREAM_PROPERTY_FLIP',
                1644167169: 'AXONLINK_STREAM_PROPERTY_CROPPING',
                1660944385: 'AXONLINK_STREAM_PROPERTY_CALIBRATION',
                1677721601: 'AXONLINK_STREAM_PROPERTY_MOTIONTHRESHOLD',
                1694498817: 'AXONLINK_STREAM_PROPERTY_EXPOSURE_LEVEL'

                }
    AXONLINK_DEVICE_PROPERTY_GET_SOFTWARE_VERSION = 1610612737
    AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS = 1610612752
    AXONLINK_DEVICE_INVOKE_SET_REBOOT = 1610612864
    AXONLINK_DEVICE_INVOKE_GET_FWVERSION = 1610613088
    AXONLINK_DEVICE_INVOKE_GET_UPGRADE_STATUS = 1610612768
    AXONLINK_DEVICE_INVOKE_SET_UPLOADFILE = 1610612784
    AXONLINK_DEVICE_INVOKE_SET_UPGRADE_ENABLE = 1610612800
    AXONLINK_DEVICE_INVOKE_GET_E2PROM_ITEM = 1610613104
    AXONLINK_DEVICE_INVOKE_GET_REGE2_STATUS = 1610612880
    AXONLINK_DEVICE_INVOKE_SET_REGE2FILE = 1610612992
    AXONLINK_DEVICE_INVOKE_SET_REGE2ENABLE = 1610613008
    AXONLINK_DEVICE_INVOKE_SET_DSP_DPINTERFACE = 1610613024
    AXONLINK_DEVICE_INVOKE_GET_DSP_DPINTERFACE = 1610613040
    AXONLINK_DEVICE_INVOKE_SET_CAMERA_TRIGER_SYNCSIGNAL = 1610613120
    AXONLINK_DEVICE_INVOKE_SET_EXPOSURE_WINDOW = 1610613136

    AXONLINK_DEVICE_INVOKE_EXTEND_BASE = 1342177280
    AXONLINK_DEVICE_INVOKE_EXTEND_MANUALEXPOSURE = 1342177281
    AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_SHUTTER = 1342177282
    AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_SHUTTER = 1342177283
    AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_GAIN = 1342177284
    AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_GAIN = 1342177285

    AXONLINK_DEVICE_COLOR_SENSOR_I2C = 1610612816
    AXONLINK_DEVICE_DEPTH_SENSOR_I2C = 1610612832
    AXONLINK_DEVICE_E2PROM = 1610612848

    AXONLINK_STREAM_PROPERTY_FLIP = 1627389953
    AXONLINK_STREAM_PROPERTY_CROPPING = 1644167169
    AXONLINK_STREAM_PROPERTY_CALIBRATION = 1660944385
    AXONLINK_STREAM_PROPERTY_MOTIONTHRESHOLD = 1677721601
    AXONLINK_STREAM_PROPERTY_EXPOSURE_LEVEL = 1694498817


AXONLINK_DEVICE_PROPERTY_GET_SOFTWARE_VERSION = AXONLINK.AXONLINK_DEVICE_PROPERTY_GET_SOFTWARE_VERSION
AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS = AXONLINK.AXONLINK_DEVICE_PROPERTY_GET_CAMERA_PARAMETERS
AXONLINK_DEVICE_INVOKE_SET_REBOOT = AXONLINK.AXONLINK_DEVICE_INVOKE_SET_REBOOT
AXONLINK_DEVICE_INVOKE_GET_FWVERSION = AXONLINK.AXONLINK_DEVICE_INVOKE_GET_FWVERSION
AXONLINK_DEVICE_INVOKE_GET_UPGRADE_STATUS = AXONLINK.AXONLINK_DEVICE_INVOKE_GET_UPGRADE_STATUS
AXONLINK_DEVICE_INVOKE_SET_UPLOADFILE = AXONLINK.AXONLINK_DEVICE_INVOKE_SET_UPLOADFILE
AXONLINK_DEVICE_INVOKE_SET_UPGRADE_ENABLE = AXONLINK.AXONLINK_DEVICE_INVOKE_SET_UPGRADE_ENABLE
AXONLINK_DEVICE_INVOKE_GET_E2PROM_ITEM = AXONLINK.AXONLINK_DEVICE_INVOKE_GET_E2PROM_ITEM
AXONLINK_DEVICE_INVOKE_GET_REGE2_STATUS = AXONLINK.AXONLINK_DEVICE_INVOKE_GET_REGE2_STATUS
AXONLINK_DEVICE_INVOKE_SET_REGE2FILE = AXONLINK.AXONLINK_DEVICE_INVOKE_SET_REGE2FILE
AXONLINK_DEVICE_INVOKE_SET_REGE2ENABLE = AXONLINK.AXONLINK_DEVICE_INVOKE_SET_REGE2ENABLE
AXONLINK_DEVICE_INVOKE_SET_DSP_DPINTERFACE = AXONLINK.AXONLINK_DEVICE_INVOKE_SET_DSP_DPINTERFACE
AXONLINK_DEVICE_INVOKE_GET_DSP_DPINTERFACE = AXONLINK.AXONLINK_DEVICE_INVOKE_GET_DSP_DPINTERFACE
AXONLINK_DEVICE_INVOKE_SET_CAMERA_TRIGER_SYNCSIGNAL = AXONLINK.AXONLINK_DEVICE_INVOKE_SET_CAMERA_TRIGER_SYNCSIGNAL
AXONLINK_DEVICE_INVOKE_SET_EXPOSURE_WINDOW = AXONLINK.AXONLINK_DEVICE_INVOKE_SET_EXPOSURE_WINDOW

AXONLINK_DEVICE_INVOKE_EXTEND_BASE = AXONLINK.AXONLINK_DEVICE_INVOKE_EXTEND_BASE
AXONLINK_DEVICE_INVOKE_EXTEND_MANUALEXPOSURE = AXONLINK.AXONLINK_DEVICE_INVOKE_EXTEND_MANUALEXPOSURE
AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_SHUTTER = AXONLINK.AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_SHUTTER
AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_SHUTTER = AXONLINK.AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_SHUTTER
AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_GAIN = AXONLINK.AXONLINK_DEVICE_INVOKE_EXTEND_SET_COLOR_GAIN
AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_GAIN = AXONLINK.AXONLINK_DEVICE_INVOKE_EXTEND_GET_COLOR_GAIN

AXONLINK_DEVICE_COLOR_SENSOR_I2C = AXONLINK.AXONLINK_DEVICE_COLOR_SENSOR_I2C
AXONLINK_DEVICE_DEPTH_SENSOR_I2C = AXONLINK.AXONLINK_DEVICE_DEPTH_SENSOR_I2C
AXONLINK_DEVICE_E2PROM = AXONLINK.AXONLINK_DEVICE_E2PROM

AXONLINK_STREAM_PROPERTY_FLIP = AXONLINK.AXONLINK_STREAM_PROPERTY_FLIP
AXONLINK_STREAM_PROPERTY_CROPPING = AXONLINK.AXONLINK_STREAM_PROPERTY_CROPPING
AXONLINK_STREAM_PROPERTY_CALIBRATION = AXONLINK.AXONLINK_STREAM_PROPERTY_CALIBRATION
AXONLINK_STREAM_PROPERTY_MOTIONTHRESHOLD = AXONLINK.AXONLINK_STREAM_PROPERTY_MOTIONTHRESHOLD
AXONLINK_STREAM_PROPERTY_EXPOSURE_LEVEL = AXONLINK.AXONLINK_STREAM_PROPERTY_EXPOSURE_LEVEL


class AXonLinkSendFileStatus(CEnum):
    _names_ = {'AXON_LINK_SENDFILE_STATUS_STOP': 0,
               'AXON_LINK_SENDFILE_STATUS_READY': 1,
               'AXON_LINK_SENDFILE_STATUS_RECVING': 2,
               'AXON_LINK_SENDFILE_STATUS_WRITING': 3,
               'AXON_LINK_SENDFILE_STATUS_SUCCESS': 4,
               'AXON_LINK_SENDFILE_STATUS_FAILED': 5
               }
    _values_ = {0: 'AXON_LINK_SENDFILE_STATUS_STOP',
                1: 'AXON_LINK_SENDFILE_STATUS_READY',
                2: 'AXON_LINK_SENDFILE_STATUS_RECVING',
                3: 'AXON_LINK_SENDFILE_STATUS_WRITING',
                4: 'AXON_LINK_SENDFILE_STATUS_SUCCESS',
                5: 'AXON_LINK_SENDFILE_STATUS_FAILED'
                }
    AXON_LINK_SENDFILE_STATUS_STOP = 0
    AXON_LINK_SENDFILE_STATUS_READY = 1
    AXON_LINK_SENDFILE_STATUS_RECVING = 2
    AXON_LINK_SENDFILE_STATUS_WRITING = 3
    AXON_LINK_SENDFILE_STATUS_SUCCESS = 4
    AXON_LINK_SENDFILE_STATUS_FAILED = 5

AXON_LINK_SENDFILE_STATUS_STOP = AXonLinkSendFileStatus.AXON_LINK_SENDFILE_STATUS_STOP
AXON_LINK_SENDFILE_STATUS_READY = AXonLinkSendFileStatus.AXON_LINK_SENDFILE_STATUS_READY
AXON_LINK_SENDFILE_STATUS_RECVING = AXonLinkSendFileStatus.AXON_LINK_SENDFILE_STATUS_RECVING
AXON_LINK_SENDFILE_STATUS_WRITING = AXonLinkSendFileStatus.AXON_LINK_SENDFILE_STATUS_WRITING
AXON_LINK_SENDFILE_STATUS_SUCCESS = AXonLinkSendFileStatus.AXON_LINK_SENDFILE_STATUS_SUCCESS
AXON_LINK_SENDFILE_STATUS_FAILED = AXonLinkSendFileStatus.AXON_LINK_SENDFILE_STATUS_FAILED



class _anon_enum_16(CEnum):
    _names_ = {'ONI_DEVICE_PROPERTY_PLAYBACK_SPEED': 100,
               'ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION': 5,
               'ONI_DEVICE_PROPERTY_DRIVER_VERSION': 1,
               'ONI_DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED': 101,
               'ONI_DEVICE_PROPERTY_ERROR_STATE': 4,
               'ONI_DEVICE_PROPERTY_HARDWARE_VERSION': 2,
               'ONI_DEVICE_PROPERTY_FIRMWARE_VERSION': 0,
               'ONI_DEVICE_PROPERTY_SERIAL_NUMBER': 3}
    _values_ = {0: 'ONI_DEVICE_PROPERTY_FIRMWARE_VERSION',
                1: 'ONI_DEVICE_PROPERTY_DRIVER_VERSION',
                2: 'ONI_DEVICE_PROPERTY_HARDWARE_VERSION',
                3: 'ONI_DEVICE_PROPERTY_SERIAL_NUMBER',
                4: 'ONI_DEVICE_PROPERTY_ERROR_STATE',
                5: 'ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION',
                100: 'ONI_DEVICE_PROPERTY_PLAYBACK_SPEED',
                101: 'ONI_DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED'
                }
    ONI_DEVICE_PROPERTY_FIRMWARE_VERSION = 0
    ONI_DEVICE_PROPERTY_DRIVER_VERSION = 1
    ONI_DEVICE_PROPERTY_HARDWARE_VERSION = 2
    ONI_DEVICE_PROPERTY_SERIAL_NUMBER = 3
    ONI_DEVICE_PROPERTY_ERROR_STATE = 4
    ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION = 5
    ONI_DEVICE_PROPERTY_PLAYBACK_SPEED = 100
    ONI_DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED = 101


ONI_DEVICE_PROPERTY_FIRMWARE_VERSION = _anon_enum_16.ONI_DEVICE_PROPERTY_FIRMWARE_VERSION
ONI_DEVICE_PROPERTY_DRIVER_VERSION = _anon_enum_16.ONI_DEVICE_PROPERTY_DRIVER_VERSION
ONI_DEVICE_PROPERTY_HARDWARE_VERSION = _anon_enum_16.ONI_DEVICE_PROPERTY_HARDWARE_VERSION
ONI_DEVICE_PROPERTY_SERIAL_NUMBER = _anon_enum_16.ONI_DEVICE_PROPERTY_SERIAL_NUMBER
ONI_DEVICE_PROPERTY_ERROR_STATE = _anon_enum_16.ONI_DEVICE_PROPERTY_ERROR_STATE
ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION = _anon_enum_16.ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION
ONI_DEVICE_PROPERTY_PLAYBACK_SPEED = _anon_enum_16.ONI_DEVICE_PROPERTY_PLAYBACK_SPEED
ONI_DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED = _anon_enum_16.ONI_DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED


class _anon_enum_17(CEnum):
    _names_ = {'ONI_STREAM_PROPERTY_MIRRORING': 7,
               'ONI_STREAM_PROPERTY_GAIN': 103,
               'ONI_STREAM_PROPERTY_NUMBER_OF_FRAMES': 8,
               'ONI_STREAM_PROPERTY_AUTO_EXPOSURE': 101,
               'ONI_STREAM_PROPERTY_STRIDE': 6,
               'ONI_STREAM_PROPERTY_HORIZONTAL_FOV': 1,
               'ONI_STREAM_PROPERTY_MIN_VALUE': 5,
               'ONI_STREAM_PROPERTY_VERTICAL_FOV': 2,
               'ONI_STREAM_PROPERTY_MAX_VALUE': 4,
               'ONI_STREAM_PROPERTY_CROPPING': 0,
               'ONI_STREAM_PROPERTY_EXPOSURE': 102,
               'ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE': 100,
               'ONI_STREAM_PROPERTY_VIDEO_MODE': 3}
    _values_ = {0: 'ONI_STREAM_PROPERTY_CROPPING',
                1: 'ONI_STREAM_PROPERTY_HORIZONTAL_FOV',
                2: 'ONI_STREAM_PROPERTY_VERTICAL_FOV',
                3: 'ONI_STREAM_PROPERTY_VIDEO_MODE',
                4: 'ONI_STREAM_PROPERTY_MAX_VALUE',
                5: 'ONI_STREAM_PROPERTY_MIN_VALUE',
                6: 'ONI_STREAM_PROPERTY_STRIDE',
                7: 'ONI_STREAM_PROPERTY_MIRRORING',
                8: 'ONI_STREAM_PROPERTY_NUMBER_OF_FRAMES',
                103: 'ONI_STREAM_PROPERTY_GAIN',
                100: 'ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE',
                102: 'ONI_STREAM_PROPERTY_EXPOSURE',
                101: 'ONI_STREAM_PROPERTY_AUTO_EXPOSURE'}
    ONI_STREAM_PROPERTY_CROPPING = 0
    ONI_STREAM_PROPERTY_HORIZONTAL_FOV = 1
    ONI_STREAM_PROPERTY_VERTICAL_FOV = 2
    ONI_STREAM_PROPERTY_VIDEO_MODE = 3
    ONI_STREAM_PROPERTY_MAX_VALUE = 4
    ONI_STREAM_PROPERTY_MIN_VALUE = 5
    ONI_STREAM_PROPERTY_STRIDE = 6
    ONI_STREAM_PROPERTY_MIRRORING = 7
    ONI_STREAM_PROPERTY_NUMBER_OF_FRAMES = 8
    ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE = 100
    ONI_STREAM_PROPERTY_AUTO_EXPOSURE = 101
    ONI_STREAM_PROPERTY_EXPOSURE = 102
    ONI_STREAM_PROPERTY_GAIN = 103


ONI_STREAM_PROPERTY_CROPPING = _anon_enum_17.ONI_STREAM_PROPERTY_CROPPING
ONI_STREAM_PROPERTY_HORIZONTAL_FOV = _anon_enum_17.ONI_STREAM_PROPERTY_HORIZONTAL_FOV
ONI_STREAM_PROPERTY_VERTICAL_FOV = _anon_enum_17.ONI_STREAM_PROPERTY_VERTICAL_FOV
ONI_STREAM_PROPERTY_VIDEO_MODE = _anon_enum_17.ONI_STREAM_PROPERTY_VIDEO_MODE
ONI_STREAM_PROPERTY_MAX_VALUE = _anon_enum_17.ONI_STREAM_PROPERTY_MAX_VALUE
ONI_STREAM_PROPERTY_MIN_VALUE = _anon_enum_17.ONI_STREAM_PROPERTY_MIN_VALUE
ONI_STREAM_PROPERTY_STRIDE = _anon_enum_17.ONI_STREAM_PROPERTY_STRIDE
ONI_STREAM_PROPERTY_MIRRORING = _anon_enum_17.ONI_STREAM_PROPERTY_MIRRORING
ONI_STREAM_PROPERTY_NUMBER_OF_FRAMES = _anon_enum_17.ONI_STREAM_PROPERTY_NUMBER_OF_FRAMES
ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE = _anon_enum_17.ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE
ONI_STREAM_PROPERTY_AUTO_EXPOSURE = _anon_enum_17.ONI_STREAM_PROPERTY_AUTO_EXPOSURE
ONI_STREAM_PROPERTY_EXPOSURE = _anon_enum_17.ONI_STREAM_PROPERTY_EXPOSURE
ONI_STREAM_PROPERTY_GAIN = _anon_enum_17.ONI_STREAM_PROPERTY_GAIN


class _anon_enum_18(CEnum):
    _names_ = {'ONI_DEVICE_COMMAND_SEEK': 1}
    _values_ = {1: 'ONI_DEVICE_COMMAND_SEEK'}
    ONI_DEVICE_COMMAND_SEEK = 1


ONI_DEVICE_COMMAND_SEEK = _anon_enum_18.ONI_DEVICE_COMMAND_SEEK


class _anon_enum_19(CEnum):
    _names_ = {'XN_STREAM_PROPERTY_REGISTRATION_TYPE': 276828165,
               'XN_MODULE_PROPERTY_FIRMWARE_FRAME_SYNC': 276885512,
               'XN_MODULE_PROPERTY_MIRROR': 276885506,
               'XN_STREAM_PROPERTY_PARAM_COEFF': 276828170,
               'XN_MODULE_PROPERTY_FIRMWARE_PARAM': 276881409,
               'XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION': 276889414,
               'XN_MODULE_PROPERTY_FIRMWARE_CPU_INTERVAL': 276889475,
               'XN_MODULE_PROPERTY_FILE': 276889478,
               'XN_MODULE_PROPERTY_SERIAL_NUMBER': 276885510,
               'XN_MODULE_PROPERTY_IMAGE_CONTROL': 276881411,
               'XN_STREAM_PROPERTY_FLICKER': 276832257,
               'XN_MODULE_PROPERTY_FILE_ATTRIBUTES': 276889480,
               'XN_MODULE_PROPERTY_PRINT_FIRMWARE_LOG': 276889472,
               'XN_MODULE_PROPERTY_TEC_SET_POINT': 276889481,
               'XN_MODULE_PROPERTY_FIRMWARE_LOG_FILTER': 276889473,
               'XN_STREAM_PROPERTY_PIXEL_REGISTRATION': 276828161,
               'XN_MODULE_PROPERTY_SENSOR_PLATFORM_STRING': 276889468,
               'XN_STREAM_PROPERTY_S2D_TABLE': 276828176,
               'XN_MODULE_PROPERTY_CMOS_BLANKING_UNITS': 276889460,
               'XN_STREAM_PROPERTY_AGC_BIN': 276828166,
               'XN_MODULE_PROPERTY_EMITTER_SET_POINT': 276889484,
               'XN_MODULE_PROPERTY_EMITTER_STATUS': 276889485,
               'XN_STREAM_PROPERTY_GMC_DEBUG': 276889413,
               'XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION_DEBUG': 276889415,
               'XN_MODULE_PROPERTY_EMITTER_STATE': 276881415,
               'XN_MODULE_PROPERTY_DEPTH_CONTROL': 276881412,
               'XN_STREAM_PROPERTY_DEPTH_SENSOR_CALIBRATION_INFO': 276828178,
               'XN_STREAM_PROPERTY_CONST_SHIFT': 276828167,
               'XN_STREAM_PROPERTY_DCMOS_RCMOS_DISTANCE': 276828175,
               'XN_STREAM_PROPERTY_MAX_SHIFT': 276828169,
               'XN_MODULE_PROPERTY_I2C': 276889486,
               'XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE': 276828174,
               'XN_MODULE_PROPERTY_RESET': 276881410,
               'XN_STREAM_PROPERTY_GMC_MODE': 276889412,
               'XN_MODULE_PROPERTY_RESET_SENSOR_ON_STARTUP': 276885508,
               'XN_MODULE_PROPERTY_CLOSE_STREAMS_ON_SHUTDOWN': 276889464,
               'XN_MODULE_PROPERTY_FIRMWARE_LOG_INTERVAL': 276889471,
               'XN_STREAM_PROPERTY_INPUT_FORMAT': 276824065,
               'XN_MODULE_PROPERTY_TEC_FAST_CONVERGENCE_STATUS': 276889483,
               'XN_MODULE_PROPERTY_PROJECTOR_FAULT': 276889488,
               'XN_MODULE_PROPERTY_CMOS_BLANKING_TIME': 276889461,
               'XN_MODULE_PROPERTY_PHYSICAL_DEVICE_NAME': 276889466,
               'XN_MODULE_PROPERTY_HOST_TIMESTAMPS': 276889463,
               'XN_STREAM_PROPERTY_SHIFT_SCALE': 276828171,
               'XN_STREAM_PROPERTY_WHITE_BALANCE_ENABLED': 276828162,
               'XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE': 276828173,
               'XN_MODULE_PROPERTY_AHB': 276881413,
               'XN_MODULE_PROPERTY_USB_INTERFACE': 276885505,
               'XN_STREAM_PROPERTY_CLOSE_RANGE': 276885507,
               'XN_MODULE_PROPERTY_BIST': 276889487,
               'XN_MODULE_PROPERTY_LED_STATE': 276881414,
               'XN_MODULE_PROPERTY_FLASH_CHUNK': 276889477,
               'XN_STREAM_PROPERTY_D2S_TABLE': 276828177,
               'XN_MODULE_PROPERTY_FILE_LIST': 276889476,
               'XN_STREAM_PROPERTY_GAIN': 276828163,
               'XN_STREAM_PROPERTY_CROPPING_MODE': 276824066,
               'XN_STREAM_PROPERTY_HOLE_FILTER': 276828164,
               'XN_MODULE_PROPERTY_VERSION': 276885511,
               'XN_MODULE_PROPERTY_DELETE_FILE': 276889479,
               'XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE': 276828172,
               'XN_MODULE_PROPERTY_TEC_STATUS': 276889482,
               'XN_MODULE_PROPERTY_FIRMWARE_TEC_DEBUG_PRINT': 276889490,
               'XN_MODULE_PROPERTY_VENDOR_SPECIFIC_DATA': 276889467,
               'XN_MODULE_PROPERTY_LEAN_INIT': 276885509,
               'XN_MODULE_PROPERTY_APC_ENABLED': 276889489,
               'XN_MODULE_PROPERTY_FIRMWARE_LOG': 276889474,
               'XN_STREAM_PROPERTY_PIXEL_SIZE_FACTOR': 276828168}

    _values_ = {276889472: 'XN_MODULE_PROPERTY_PRINT_FIRMWARE_LOG',
                276885505: 'XN_MODULE_PROPERTY_USB_INTERFACE',
                276885506: 'XN_MODULE_PROPERTY_MIRROR',
                276889475: 'XN_MODULE_PROPERTY_FIRMWARE_CPU_INTERVAL',
                276885508: 'XN_MODULE_PROPERTY_RESET_SENSOR_ON_STARTUP',
                276885509: 'XN_MODULE_PROPERTY_LEAN_INIT',
                276885510: 'XN_MODULE_PROPERTY_SERIAL_NUMBER',
                276885511: 'XN_MODULE_PROPERTY_VERSION',
                276885512: 'XN_MODULE_PROPERTY_FIRMWARE_FRAME_SYNC',
                276889481: 'XN_MODULE_PROPERTY_TEC_SET_POINT',
                276889482: 'XN_MODULE_PROPERTY_TEC_STATUS',
                276889483: 'XN_MODULE_PROPERTY_TEC_FAST_CONVERGENCE_STATUS',
                276889484: 'XN_MODULE_PROPERTY_EMITTER_SET_POINT',
                276889474: 'XN_MODULE_PROPERTY_FIRMWARE_LOG',
                276889486: 'XN_MODULE_PROPERTY_I2C',
                276889487: 'XN_MODULE_PROPERTY_BIST',
                276889488: 'XN_MODULE_PROPERTY_PROJECTOR_FAULT',
                276889489: 'XN_MODULE_PROPERTY_APC_ENABLED',
                276889490: 'XN_MODULE_PROPERTY_FIRMWARE_TEC_DEBUG_PRINT',
                276881411: 'XN_MODULE_PROPERTY_IMAGE_CONTROL',
                276889413: 'XN_STREAM_PROPERTY_GMC_DEBUG',
                276889477: 'XN_MODULE_PROPERTY_FLASH_CHUNK',
                276881412: 'XN_MODULE_PROPERTY_DEPTH_CONTROL',
                276828165: 'XN_STREAM_PROPERTY_REGISTRATION_TYPE',
                276881413: 'XN_MODULE_PROPERTY_AHB',
                276889473: 'XN_MODULE_PROPERTY_FIRMWARE_LOG_FILTER',
                276889476: 'XN_MODULE_PROPERTY_FILE_LIST',
                276881409: 'XN_MODULE_PROPERTY_FIRMWARE_PARAM',
                276881414: 'XN_MODULE_PROPERTY_LED_STATE',
                276881415: 'XN_MODULE_PROPERTY_EMITTER_STATE',
                276889480: 'XN_MODULE_PROPERTY_FILE_ATTRIBUTES',
                276832257: 'XN_STREAM_PROPERTY_FLICKER',
                276824065: 'XN_STREAM_PROPERTY_INPUT_FORMAT',
                276889478: 'XN_MODULE_PROPERTY_FILE',
                276828169: 'XN_STREAM_PROPERTY_MAX_SHIFT',
                276828161: 'XN_STREAM_PROPERTY_PIXEL_REGISTRATION',
                276828166: 'XN_STREAM_PROPERTY_AGC_BIN',
                276828170: 'XN_STREAM_PROPERTY_PARAM_COEFF',
                276828173: 'XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE',
                276881410: 'XN_MODULE_PROPERTY_RESET',
                276828171: 'XN_STREAM_PROPERTY_SHIFT_SCALE',
                276889412: 'XN_STREAM_PROPERTY_GMC_MODE',
                276828162: 'XN_STREAM_PROPERTY_WHITE_BALANCE_ENABLED',
                276889414: 'XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION',
                276889415: 'XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION_DEBUG',
                276828172: 'XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE',
                276824066: 'XN_STREAM_PROPERTY_CROPPING_MODE',
                276889485: 'XN_MODULE_PROPERTY_EMITTER_STATUS',
                276889479: 'XN_MODULE_PROPERTY_DELETE_FILE',
                276828174: 'XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE',
                276828167: 'XN_STREAM_PROPERTY_CONST_SHIFT',
                276828175: 'XN_STREAM_PROPERTY_DCMOS_RCMOS_DISTANCE',
                276885507: 'XN_STREAM_PROPERTY_CLOSE_RANGE',
                276828176: 'XN_STREAM_PROPERTY_S2D_TABLE',
                276828163: 'XN_STREAM_PROPERTY_GAIN',
                276828177: 'XN_STREAM_PROPERTY_D2S_TABLE',
                276828178: 'XN_STREAM_PROPERTY_DEPTH_SENSOR_CALIBRATION_INFO',
                276889460: 'XN_MODULE_PROPERTY_CMOS_BLANKING_UNITS',
                276889461: 'XN_MODULE_PROPERTY_CMOS_BLANKING_TIME',
                276828168: 'XN_STREAM_PROPERTY_PIXEL_SIZE_FACTOR',
                276889463: 'XN_MODULE_PROPERTY_HOST_TIMESTAMPS',
                276889464: 'XN_MODULE_PROPERTY_CLOSE_STREAMS_ON_SHUTDOWN',
                276889466: 'XN_MODULE_PROPERTY_PHYSICAL_DEVICE_NAME',
                276889467: 'XN_MODULE_PROPERTY_VENDOR_SPECIFIC_DATA',
                276889468: 'XN_MODULE_PROPERTY_SENSOR_PLATFORM_STRING',
                276828164: 'XN_STREAM_PROPERTY_HOLE_FILTER',
                276889471: 'XN_MODULE_PROPERTY_FIRMWARE_LOG_INTERVAL'}
    XN_MODULE_PROPERTY_USB_INTERFACE = 276885505
    XN_MODULE_PROPERTY_MIRROR = 276885506
    XN_MODULE_PROPERTY_RESET_SENSOR_ON_STARTUP = 276885508
    XN_MODULE_PROPERTY_LEAN_INIT = 276885509
    XN_MODULE_PROPERTY_SERIAL_NUMBER = 276885510
    XN_MODULE_PROPERTY_VERSION = 276885511
    XN_MODULE_PROPERTY_FIRMWARE_FRAME_SYNC = 276885512
    XN_MODULE_PROPERTY_HOST_TIMESTAMPS = 276889463
    XN_MODULE_PROPERTY_CLOSE_STREAMS_ON_SHUTDOWN = 276889464
    XN_MODULE_PROPERTY_FIRMWARE_LOG_INTERVAL = 276889471
    XN_MODULE_PROPERTY_PRINT_FIRMWARE_LOG = 276889472
    XN_MODULE_PROPERTY_FIRMWARE_LOG_FILTER = 276889473
    XN_MODULE_PROPERTY_FIRMWARE_LOG = 276889474
    XN_MODULE_PROPERTY_FIRMWARE_CPU_INTERVAL = 276889475
    XN_MODULE_PROPERTY_PHYSICAL_DEVICE_NAME = 276889466
    XN_MODULE_PROPERTY_VENDOR_SPECIFIC_DATA = 276889467
    XN_MODULE_PROPERTY_SENSOR_PLATFORM_STRING = 276889468
    XN_MODULE_PROPERTY_FIRMWARE_PARAM = 276881409
    XN_MODULE_PROPERTY_RESET = 276881410
    XN_MODULE_PROPERTY_IMAGE_CONTROL = 276881411
    XN_MODULE_PROPERTY_DEPTH_CONTROL = 276881412
    XN_MODULE_PROPERTY_AHB = 276881413
    XN_MODULE_PROPERTY_LED_STATE = 276881414
    XN_MODULE_PROPERTY_EMITTER_STATE = 276881415
    XN_MODULE_PROPERTY_CMOS_BLANKING_UNITS = 276889460
    XN_MODULE_PROPERTY_CMOS_BLANKING_TIME = 276889461
    XN_MODULE_PROPERTY_FILE_LIST = 276889476
    XN_MODULE_PROPERTY_FLASH_CHUNK = 276889477
    XN_MODULE_PROPERTY_FILE = 276889478
    XN_MODULE_PROPERTY_DELETE_FILE = 276889479
    XN_MODULE_PROPERTY_FILE_ATTRIBUTES = 276889480
    XN_MODULE_PROPERTY_TEC_SET_POINT = 276889481
    XN_MODULE_PROPERTY_TEC_STATUS = 276889482
    XN_MODULE_PROPERTY_TEC_FAST_CONVERGENCE_STATUS = 276889483
    XN_MODULE_PROPERTY_EMITTER_SET_POINT = 276889484
    XN_MODULE_PROPERTY_EMITTER_STATUS = 276889485
    XN_MODULE_PROPERTY_I2C = 276889486
    XN_MODULE_PROPERTY_BIST = 276889487
    XN_MODULE_PROPERTY_PROJECTOR_FAULT = 276889488
    XN_MODULE_PROPERTY_APC_ENABLED = 276889489
    XN_MODULE_PROPERTY_FIRMWARE_TEC_DEBUG_PRINT = 276889490
    XN_STREAM_PROPERTY_INPUT_FORMAT = 276824065
    XN_STREAM_PROPERTY_CROPPING_MODE = 276824066
    XN_STREAM_PROPERTY_CLOSE_RANGE = 276885507
    XN_STREAM_PROPERTY_PIXEL_REGISTRATION = 276828161
    XN_STREAM_PROPERTY_WHITE_BALANCE_ENABLED = 276828162
    XN_STREAM_PROPERTY_GAIN = 276828163
    XN_STREAM_PROPERTY_HOLE_FILTER = 276828164
    XN_STREAM_PROPERTY_REGISTRATION_TYPE = 276828165
    XN_STREAM_PROPERTY_AGC_BIN = 276828166
    XN_STREAM_PROPERTY_CONST_SHIFT = 276828167
    XN_STREAM_PROPERTY_PIXEL_SIZE_FACTOR = 276828168
    XN_STREAM_PROPERTY_MAX_SHIFT = 276828169
    XN_STREAM_PROPERTY_PARAM_COEFF = 276828170
    XN_STREAM_PROPERTY_SHIFT_SCALE = 276828171
    XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE = 276828172
    XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE = 276828173
    XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE = 276828174
    XN_STREAM_PROPERTY_DCMOS_RCMOS_DISTANCE = 276828175
    XN_STREAM_PROPERTY_S2D_TABLE = 276828176
    XN_STREAM_PROPERTY_D2S_TABLE = 276828177
    XN_STREAM_PROPERTY_DEPTH_SENSOR_CALIBRATION_INFO = 276828178
    XN_STREAM_PROPERTY_GMC_MODE = 276889412
    XN_STREAM_PROPERTY_GMC_DEBUG = 276889413
    XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION = 276889414
    XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION_DEBUG = 276889415
    XN_STREAM_PROPERTY_FLICKER = 276832257


XN_MODULE_PROPERTY_USB_INTERFACE = _anon_enum_19.XN_MODULE_PROPERTY_USB_INTERFACE
XN_MODULE_PROPERTY_MIRROR = _anon_enum_19.XN_MODULE_PROPERTY_MIRROR
XN_MODULE_PROPERTY_RESET_SENSOR_ON_STARTUP = _anon_enum_19.XN_MODULE_PROPERTY_RESET_SENSOR_ON_STARTUP
XN_MODULE_PROPERTY_LEAN_INIT = _anon_enum_19.XN_MODULE_PROPERTY_LEAN_INIT
XN_MODULE_PROPERTY_SERIAL_NUMBER = _anon_enum_19.XN_MODULE_PROPERTY_SERIAL_NUMBER
XN_MODULE_PROPERTY_VERSION = _anon_enum_19.XN_MODULE_PROPERTY_VERSION
XN_MODULE_PROPERTY_FIRMWARE_FRAME_SYNC = _anon_enum_19.XN_MODULE_PROPERTY_FIRMWARE_FRAME_SYNC
XN_MODULE_PROPERTY_HOST_TIMESTAMPS = _anon_enum_19.XN_MODULE_PROPERTY_HOST_TIMESTAMPS
XN_MODULE_PROPERTY_CLOSE_STREAMS_ON_SHUTDOWN = _anon_enum_19.XN_MODULE_PROPERTY_CLOSE_STREAMS_ON_SHUTDOWN
XN_MODULE_PROPERTY_FIRMWARE_LOG_INTERVAL = _anon_enum_19.XN_MODULE_PROPERTY_FIRMWARE_LOG_INTERVAL
XN_MODULE_PROPERTY_PRINT_FIRMWARE_LOG = _anon_enum_19.XN_MODULE_PROPERTY_PRINT_FIRMWARE_LOG
XN_MODULE_PROPERTY_FIRMWARE_LOG_FILTER = _anon_enum_19.XN_MODULE_PROPERTY_FIRMWARE_LOG_FILTER
XN_MODULE_PROPERTY_FIRMWARE_LOG = _anon_enum_19.XN_MODULE_PROPERTY_FIRMWARE_LOG
XN_MODULE_PROPERTY_FIRMWARE_CPU_INTERVAL = _anon_enum_19.XN_MODULE_PROPERTY_FIRMWARE_CPU_INTERVAL
XN_MODULE_PROPERTY_PHYSICAL_DEVICE_NAME = _anon_enum_19.XN_MODULE_PROPERTY_PHYSICAL_DEVICE_NAME
XN_MODULE_PROPERTY_VENDOR_SPECIFIC_DATA = _anon_enum_19.XN_MODULE_PROPERTY_VENDOR_SPECIFIC_DATA
XN_MODULE_PROPERTY_SENSOR_PLATFORM_STRING = _anon_enum_19.XN_MODULE_PROPERTY_SENSOR_PLATFORM_STRING
XN_MODULE_PROPERTY_FIRMWARE_PARAM = _anon_enum_19.XN_MODULE_PROPERTY_FIRMWARE_PARAM
XN_MODULE_PROPERTY_RESET = _anon_enum_19.XN_MODULE_PROPERTY_RESET
XN_MODULE_PROPERTY_IMAGE_CONTROL = _anon_enum_19.XN_MODULE_PROPERTY_IMAGE_CONTROL
XN_MODULE_PROPERTY_DEPTH_CONTROL = _anon_enum_19.XN_MODULE_PROPERTY_DEPTH_CONTROL
XN_MODULE_PROPERTY_AHB = _anon_enum_19.XN_MODULE_PROPERTY_AHB
XN_MODULE_PROPERTY_LED_STATE = _anon_enum_19.XN_MODULE_PROPERTY_LED_STATE
XN_MODULE_PROPERTY_EMITTER_STATE = _anon_enum_19.XN_MODULE_PROPERTY_EMITTER_STATE
XN_MODULE_PROPERTY_CMOS_BLANKING_UNITS = _anon_enum_19.XN_MODULE_PROPERTY_CMOS_BLANKING_UNITS
XN_MODULE_PROPERTY_CMOS_BLANKING_TIME = _anon_enum_19.XN_MODULE_PROPERTY_CMOS_BLANKING_TIME
XN_MODULE_PROPERTY_FILE_LIST = _anon_enum_19.XN_MODULE_PROPERTY_FILE_LIST
XN_MODULE_PROPERTY_FLASH_CHUNK = _anon_enum_19.XN_MODULE_PROPERTY_FLASH_CHUNK
XN_MODULE_PROPERTY_FILE = _anon_enum_19.XN_MODULE_PROPERTY_FILE
XN_MODULE_PROPERTY_DELETE_FILE = _anon_enum_19.XN_MODULE_PROPERTY_DELETE_FILE
XN_MODULE_PROPERTY_FILE_ATTRIBUTES = _anon_enum_19.XN_MODULE_PROPERTY_FILE_ATTRIBUTES
XN_MODULE_PROPERTY_TEC_SET_POINT = _anon_enum_19.XN_MODULE_PROPERTY_TEC_SET_POINT
XN_MODULE_PROPERTY_TEC_STATUS = _anon_enum_19.XN_MODULE_PROPERTY_TEC_STATUS
XN_MODULE_PROPERTY_TEC_FAST_CONVERGENCE_STATUS = _anon_enum_19.XN_MODULE_PROPERTY_TEC_FAST_CONVERGENCE_STATUS
XN_MODULE_PROPERTY_EMITTER_SET_POINT = _anon_enum_19.XN_MODULE_PROPERTY_EMITTER_SET_POINT
XN_MODULE_PROPERTY_EMITTER_STATUS = _anon_enum_19.XN_MODULE_PROPERTY_EMITTER_STATUS
XN_MODULE_PROPERTY_I2C = _anon_enum_19.XN_MODULE_PROPERTY_I2C
XN_MODULE_PROPERTY_BIST = _anon_enum_19.XN_MODULE_PROPERTY_BIST
XN_MODULE_PROPERTY_PROJECTOR_FAULT = _anon_enum_19.XN_MODULE_PROPERTY_PROJECTOR_FAULT
XN_MODULE_PROPERTY_APC_ENABLED = _anon_enum_19.XN_MODULE_PROPERTY_APC_ENABLED
XN_MODULE_PROPERTY_FIRMWARE_TEC_DEBUG_PRINT = _anon_enum_19.XN_MODULE_PROPERTY_FIRMWARE_TEC_DEBUG_PRINT
XN_STREAM_PROPERTY_INPUT_FORMAT = _anon_enum_19.XN_STREAM_PROPERTY_INPUT_FORMAT
XN_STREAM_PROPERTY_CROPPING_MODE = _anon_enum_19.XN_STREAM_PROPERTY_CROPPING_MODE
XN_STREAM_PROPERTY_CLOSE_RANGE = _anon_enum_19.XN_STREAM_PROPERTY_CLOSE_RANGE
XN_STREAM_PROPERTY_PIXEL_REGISTRATION = _anon_enum_19.XN_STREAM_PROPERTY_PIXEL_REGISTRATION
XN_STREAM_PROPERTY_WHITE_BALANCE_ENABLED = _anon_enum_19.XN_STREAM_PROPERTY_WHITE_BALANCE_ENABLED
XN_STREAM_PROPERTY_GAIN = _anon_enum_19.XN_STREAM_PROPERTY_GAIN
XN_STREAM_PROPERTY_HOLE_FILTER = _anon_enum_19.XN_STREAM_PROPERTY_HOLE_FILTER
XN_STREAM_PROPERTY_REGISTRATION_TYPE = _anon_enum_19.XN_STREAM_PROPERTY_REGISTRATION_TYPE
XN_STREAM_PROPERTY_AGC_BIN = _anon_enum_19.XN_STREAM_PROPERTY_AGC_BIN
XN_STREAM_PROPERTY_CONST_SHIFT = _anon_enum_19.XN_STREAM_PROPERTY_CONST_SHIFT
XN_STREAM_PROPERTY_PIXEL_SIZE_FACTOR = _anon_enum_19.XN_STREAM_PROPERTY_PIXEL_SIZE_FACTOR
XN_STREAM_PROPERTY_MAX_SHIFT = _anon_enum_19.XN_STREAM_PROPERTY_MAX_SHIFT
XN_STREAM_PROPERTY_PARAM_COEFF = _anon_enum_19.XN_STREAM_PROPERTY_PARAM_COEFF
XN_STREAM_PROPERTY_SHIFT_SCALE = _anon_enum_19.XN_STREAM_PROPERTY_SHIFT_SCALE
XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE = _anon_enum_19.XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE
XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE = _anon_enum_19.XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE
XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE = _anon_enum_19.XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE
XN_STREAM_PROPERTY_DCMOS_RCMOS_DISTANCE = _anon_enum_19.XN_STREAM_PROPERTY_DCMOS_RCMOS_DISTANCE
XN_STREAM_PROPERTY_S2D_TABLE = _anon_enum_19.XN_STREAM_PROPERTY_S2D_TABLE
XN_STREAM_PROPERTY_D2S_TABLE = _anon_enum_19.XN_STREAM_PROPERTY_D2S_TABLE
XN_STREAM_PROPERTY_DEPTH_SENSOR_CALIBRATION_INFO = _anon_enum_19.XN_STREAM_PROPERTY_DEPTH_SENSOR_CALIBRATION_INFO
XN_STREAM_PROPERTY_GMC_MODE = _anon_enum_19.XN_STREAM_PROPERTY_GMC_MODE
XN_STREAM_PROPERTY_GMC_DEBUG = _anon_enum_19.XN_STREAM_PROPERTY_GMC_DEBUG
XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION = _anon_enum_19.XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION
XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION_DEBUG = _anon_enum_19.XN_STREAM_PROPERTY_WAVELENGTH_CORRECTION_DEBUG
XN_STREAM_PROPERTY_FLICKER = _anon_enum_19.XN_STREAM_PROPERTY_FLICKER


class XnFWVer(CEnum):
    _names_ = {'XN_SENSOR_FW_VER_5_6': 12,
               'XN_SENSOR_FW_VER_5_2': 8,
               'XN_SENSOR_FW_VER_5_0': 6,
               'XN_SENSOR_FW_VER_5_1': 7,
               'XN_SENSOR_FW_VER_0_17': 1,
               'XN_SENSOR_FW_VER_5_3': 9,
               'XN_SENSOR_FW_VER_5_4': 10,
               'XN_SENSOR_FW_VER_1_1': 2,
               'XN_SENSOR_FW_VER_1_2': 3,
               'XN_SENSOR_FW_VER_5_7': 13,
               'XN_SENSOR_FW_VER_5_8': 14,
               'XN_SENSOR_FW_VER_4_0': 5,
               'XN_SENSOR_FW_VER_UNKNOWN': 0,
               'XN_SENSOR_FW_VER_3_0': 4,
               'XN_SENSOR_FW_VER_5_5': 11}
    _values_ = {0: 'XN_SENSOR_FW_VER_UNKNOWN',
                1: 'XN_SENSOR_FW_VER_0_17',
                2: 'XN_SENSOR_FW_VER_1_1',
                3: 'XN_SENSOR_FW_VER_1_2',
                4: 'XN_SENSOR_FW_VER_3_0',
                5: 'XN_SENSOR_FW_VER_4_0',
                6: 'XN_SENSOR_FW_VER_5_0',
                7: 'XN_SENSOR_FW_VER_5_1',
                8: 'XN_SENSOR_FW_VER_5_2',
                9: 'XN_SENSOR_FW_VER_5_3',
                10: 'XN_SENSOR_FW_VER_5_4',
                11: 'XN_SENSOR_FW_VER_5_5',
                12: 'XN_SENSOR_FW_VER_5_6',
                13: 'XN_SENSOR_FW_VER_5_7',
                14: 'XN_SENSOR_FW_VER_5_8'}
    XN_SENSOR_FW_VER_UNKNOWN = 0
    XN_SENSOR_FW_VER_0_17 = 1
    XN_SENSOR_FW_VER_1_1 = 2
    XN_SENSOR_FW_VER_1_2 = 3
    XN_SENSOR_FW_VER_3_0 = 4
    XN_SENSOR_FW_VER_4_0 = 5
    XN_SENSOR_FW_VER_5_0 = 6
    XN_SENSOR_FW_VER_5_1 = 7
    XN_SENSOR_FW_VER_5_2 = 8
    XN_SENSOR_FW_VER_5_3 = 9
    XN_SENSOR_FW_VER_5_4 = 10
    XN_SENSOR_FW_VER_5_5 = 11
    XN_SENSOR_FW_VER_5_6 = 12
    XN_SENSOR_FW_VER_5_7 = 13
    XN_SENSOR_FW_VER_5_8 = 14


class XnSensorVer(CEnum):
    _names_ = {'XN_SENSOR_VER_3_0': 2,
               'XN_SENSOR_VER_4_0': 3,
               'XN_SENSOR_VER_5_0': 4,
               'XN_SENSOR_VER_2_0': 1,
               'XN_SENSOR_VER_UNKNOWN': 0}
    _values_ = {0: 'XN_SENSOR_VER_UNKNOWN',
                1: 'XN_SENSOR_VER_2_0',
                2: 'XN_SENSOR_VER_3_0',
                3: 'XN_SENSOR_VER_4_0',
                4: 'XN_SENSOR_VER_5_0'}
    XN_SENSOR_VER_UNKNOWN = 0
    XN_SENSOR_VER_2_0 = 1
    XN_SENSOR_VER_3_0 = 2
    XN_SENSOR_VER_4_0 = 3
    XN_SENSOR_VER_5_0 = 4


class XnHWVer(CEnum):
    _names_ = {'XN_SENSOR_HW_VER_RD1081': 5,
               'XN_SENSOR_HW_VER_RD1082': 6,
               'XN_SENSOR_HW_VER_FPDB_10': 1,
               'XN_SENSOR_HW_VER_CDB_10': 2,
               'XN_SENSOR_HW_VER_UNKNOWN': 0,
               'XN_SENSOR_HW_VER_RD_5': 4,
               'XN_SENSOR_HW_VER_RD109': 7,
               'XN_SENSOR_HW_VER_RD_3': 3}
    _values_ = {0: 'XN_SENSOR_HW_VER_UNKNOWN',
                1: 'XN_SENSOR_HW_VER_FPDB_10',
                2: 'XN_SENSOR_HW_VER_CDB_10',
                3: 'XN_SENSOR_HW_VER_RD_3',
                4: 'XN_SENSOR_HW_VER_RD_5',
                5: 'XN_SENSOR_HW_VER_RD1081',
                6: 'XN_SENSOR_HW_VER_RD1082',
                7: 'XN_SENSOR_HW_VER_RD109'}
    XN_SENSOR_HW_VER_UNKNOWN = 0
    XN_SENSOR_HW_VER_FPDB_10 = 1
    XN_SENSOR_HW_VER_CDB_10 = 2
    XN_SENSOR_HW_VER_RD_3 = 3
    XN_SENSOR_HW_VER_RD_5 = 4
    XN_SENSOR_HW_VER_RD1081 = 5
    XN_SENSOR_HW_VER_RD1082 = 6
    XN_SENSOR_HW_VER_RD109 = 7


class XnChipVer(CEnum):
    _names_ = {'XN_SENSOR_CHIP_VER_PS1000': 1,
               'XN_SENSOR_CHIP_VER_PS1080': 2,
               'XN_SENSOR_CHIP_VER_PS1080A6': 3,
               'XN_SENSOR_CHIP_VER_UNKNOWN': 0}
    _values_ = {0: 'XN_SENSOR_CHIP_VER_UNKNOWN',
                1: 'XN_SENSOR_CHIP_VER_PS1000',
                2: 'XN_SENSOR_CHIP_VER_PS1080',
                3: 'XN_SENSOR_CHIP_VER_PS1080A6'}
    XN_SENSOR_CHIP_VER_UNKNOWN = 0
    XN_SENSOR_CHIP_VER_PS1000 = 1
    XN_SENSOR_CHIP_VER_PS1080 = 2
    XN_SENSOR_CHIP_VER_PS1080A6 = 3


class XnCMOSType(CEnum):
    _names_ = {'XN_CMOS_TYPE_IMAGE': 0,
               'XN_CMOS_COUNT': 2,
               'XN_CMOS_TYPE_DEPTH': 1}
    _values_ = {0: 'XN_CMOS_TYPE_IMAGE',
                1: 'XN_CMOS_TYPE_DEPTH',
                2: 'XN_CMOS_COUNT'}
    XN_CMOS_TYPE_IMAGE = 0
    XN_CMOS_TYPE_DEPTH = 1
    XN_CMOS_COUNT = 2


class XnIOImageFormats(CEnum):
    _names_ = {'XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUYV': 7,
               'XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUV422': 5,
               'XN_IO_IMAGE_FORMAT_UNCOMPRESSED_BAYER': 6,
               'XN_IO_IMAGE_FORMAT_BAYER': 0,
               'XN_IO_IMAGE_FORMAT_JPEG': 2,
               'XN_IO_IMAGE_FORMAT_JPEG_420': 3,
               'XN_IO_IMAGE_FORMAT_JPEG_MONO': 4,
               'XN_IO_IMAGE_FORMAT_YUV422': 1}
    _values_ = {0: 'XN_IO_IMAGE_FORMAT_BAYER',
                1: 'XN_IO_IMAGE_FORMAT_YUV422',
                2: 'XN_IO_IMAGE_FORMAT_JPEG',
                3: 'XN_IO_IMAGE_FORMAT_JPEG_420',
                4: 'XN_IO_IMAGE_FORMAT_JPEG_MONO',
                5: 'XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUV422',
                6: 'XN_IO_IMAGE_FORMAT_UNCOMPRESSED_BAYER',
                7: 'XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUYV'}
    XN_IO_IMAGE_FORMAT_BAYER = 0
    XN_IO_IMAGE_FORMAT_YUV422 = 1
    XN_IO_IMAGE_FORMAT_JPEG = 2
    XN_IO_IMAGE_FORMAT_JPEG_420 = 3
    XN_IO_IMAGE_FORMAT_JPEG_MONO = 4
    XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUV422 = 5
    XN_IO_IMAGE_FORMAT_UNCOMPRESSED_BAYER = 6
    XN_IO_IMAGE_FORMAT_UNCOMPRESSED_YUYV = 7


class XnIODepthFormats(CEnum):
    _names_ = {'XN_IO_DEPTH_FORMAT_UNCOMPRESSED_16_BIT': 0,
               'XN_IO_DEPTH_FORMAT_COMPRESSED_PS': 1,
               'XN_IO_DEPTH_FORMAT_UNCOMPRESSED_12_BIT': 4,
               'XN_IO_DEPTH_FORMAT_UNCOMPRESSED_11_BIT': 3,
               'XN_IO_DEPTH_FORMAT_UNCOMPRESSED_10_BIT': 2}
    _values_ = {0: 'XN_IO_DEPTH_FORMAT_UNCOMPRESSED_16_BIT',
                1: 'XN_IO_DEPTH_FORMAT_COMPRESSED_PS',
                2: 'XN_IO_DEPTH_FORMAT_UNCOMPRESSED_10_BIT',
                3: 'XN_IO_DEPTH_FORMAT_UNCOMPRESSED_11_BIT',
                4: 'XN_IO_DEPTH_FORMAT_UNCOMPRESSED_12_BIT'}
    XN_IO_DEPTH_FORMAT_UNCOMPRESSED_16_BIT = 0
    XN_IO_DEPTH_FORMAT_COMPRESSED_PS = 1
    XN_IO_DEPTH_FORMAT_UNCOMPRESSED_10_BIT = 2
    XN_IO_DEPTH_FORMAT_UNCOMPRESSED_11_BIT = 3
    XN_IO_DEPTH_FORMAT_UNCOMPRESSED_12_BIT = 4


class XnParamResetType(CEnum):
    _names_ = {'XN_RESET_TYPE_POWER': 0,
               'XN_RESET_TYPE_SOFT': 1,
               'XN_RESET_TYPE_SOFT_FIRST': 2}
    _values_ = {0: 'XN_RESET_TYPE_POWER',
                1: 'XN_RESET_TYPE_SOFT',
                2: 'XN_RESET_TYPE_SOFT_FIRST'}
    XN_RESET_TYPE_POWER = 0
    XN_RESET_TYPE_SOFT = 1
    XN_RESET_TYPE_SOFT_FIRST = 2


class XnSensorUsbInterface(CEnum):
    _names_ = {'XN_SENSOR_USB_INTERFACE_ISO_ENDPOINTS': 1,
               'XN_SENSOR_USB_INTERFACE_DEFAULT': 0,
               'XN_SENSOR_USB_INTERFACE_BULK_ENDPOINTS': 2,
               'XN_SENSOR_USB_INTERFACE_ISO_ENDPOINTS_LOW_DEPTH': 3}
    _values_ = {0: 'XN_SENSOR_USB_INTERFACE_DEFAULT',
                1: 'XN_SENSOR_USB_INTERFACE_ISO_ENDPOINTS',
                2: 'XN_SENSOR_USB_INTERFACE_BULK_ENDPOINTS',
                3: 'XN_SENSOR_USB_INTERFACE_ISO_ENDPOINTS_LOW_DEPTH'}
    XN_SENSOR_USB_INTERFACE_DEFAULT = 0
    XN_SENSOR_USB_INTERFACE_ISO_ENDPOINTS = 1
    XN_SENSOR_USB_INTERFACE_BULK_ENDPOINTS = 2
    XN_SENSOR_USB_INTERFACE_ISO_ENDPOINTS_LOW_DEPTH = 3


class XnProcessingType(CEnum):
    _names_ = {'XN_PROCESSING_HARDWARE': 1,
               'XN_PROCESSING_SOFTWARE': 2,
               'XN_PROCESSING_DONT_CARE': 0}
    _values_ = {0: 'XN_PROCESSING_DONT_CARE',
                1: 'XN_PROCESSING_HARDWARE',
                2: 'XN_PROCESSING_SOFTWARE'}
    XN_PROCESSING_DONT_CARE = 0
    XN_PROCESSING_HARDWARE = 1
    XN_PROCESSING_SOFTWARE = 2


class XnCroppingMode(CEnum):
    _names_ = {'XN_CROPPING_MODE_SOFTWARE_ONLY': 3,
               'XN_CROPPING_MODE_INCREASED_FPS': 2,
               'XN_CROPPING_MODE_NORMAL': 1}
    _values_ = {1: 'XN_CROPPING_MODE_NORMAL',
                2: 'XN_CROPPING_MODE_INCREASED_FPS',
                3: 'XN_CROPPING_MODE_SOFTWARE_ONLY'}
    XN_CROPPING_MODE_NORMAL = 1
    XN_CROPPING_MODE_INCREASED_FPS = 2
    XN_CROPPING_MODE_SOFTWARE_ONLY = 3


class _anon_enum_28(CEnum):
    _names_ = {'XN_ERROR_STATE_OK': 0,
               'XN_ERROR_STATE_DEVICE_PROJECTOR_FAULT': 1,
               'XN_ERROR_STATE_DEVICE_OVERHEAT': 2}
    _values_ = {0: 'XN_ERROR_STATE_OK',
                1: 'XN_ERROR_STATE_DEVICE_PROJECTOR_FAULT',
                2: 'XN_ERROR_STATE_DEVICE_OVERHEAT'}
    XN_ERROR_STATE_OK = 0
    XN_ERROR_STATE_DEVICE_PROJECTOR_FAULT = 1
    XN_ERROR_STATE_DEVICE_OVERHEAT = 2


XN_ERROR_STATE_OK = _anon_enum_28.XN_ERROR_STATE_OK
XN_ERROR_STATE_DEVICE_PROJECTOR_FAULT = _anon_enum_28.XN_ERROR_STATE_DEVICE_PROJECTOR_FAULT
XN_ERROR_STATE_DEVICE_OVERHEAT = _anon_enum_28.XN_ERROR_STATE_DEVICE_OVERHEAT


class XnFirmwareCroppingMode(CEnum):
    _names_ = {'XN_FIRMWARE_CROPPING_MODE_DISABLED': 0,
               'XN_FIRMWARE_CROPPING_MODE_INCREASED_FPS': 2,
               'XN_FIRMWARE_CROPPING_MODE_NORMAL': 1}
    _values_ = {0: 'XN_FIRMWARE_CROPPING_MODE_DISABLED',
                1: 'XN_FIRMWARE_CROPPING_MODE_NORMAL',
                2: 'XN_FIRMWARE_CROPPING_MODE_INCREASED_FPS'}
    XN_FIRMWARE_CROPPING_MODE_DISABLED = 0
    XN_FIRMWARE_CROPPING_MODE_NORMAL = 1
    XN_FIRMWARE_CROPPING_MODE_INCREASED_FPS = 2


class XnLogFilter(CEnum):
    _names_ = {'XnLogFilterAssert': 16,
               'XnLogFilterError': 4,
               'XnLogFilterAll': 65535,
               'XnLogFilterProtocol': 8,
               'XnLogFilterTelems': 256,
               'XnLogFilterFrameSync': 64,
               'XnLogFilterInfo': 2,
               'XnLogFilterAGC': 128,
               'XnLogFilterConfig': 32,
               'XnLogFilterDebug': 1}
    _values_ = {32: 'XnLogFilterConfig',
                1: 'XnLogFilterDebug',
                2: 'XnLogFilterInfo',
                4: 'XnLogFilterError',
                8: 'XnLogFilterProtocol',
                64: 'XnLogFilterFrameSync',
                128: 'XnLogFilterAGC',
                256: 'XnLogFilterTelems',
                16: 'XnLogFilterAssert',
                65535: 'XnLogFilterAll'}
    XnLogFilterDebug = 1
    XnLogFilterInfo = 2
    XnLogFilterError = 4
    XnLogFilterProtocol = 8
    XnLogFilterAssert = 16
    XnLogFilterConfig = 32
    XnLogFilterFrameSync = 64
    XnLogFilterAGC = 128
    XnLogFilterTelems = 256
    XnLogFilterAll = 65535


class XnFilePossibleAttributes(CEnum):
    _names_ = {'XnFileAttributeReadOnly': 32768}
    _values_ = {32768: 'XnFileAttributeReadOnly'}
    XnFileAttributeReadOnly = 32768


class XnFlashFileType(CEnum):
    _names_ = {'XnFlashFileTypeAlgorithmParams': 12,
               'XnFlashFileTypeCodeDownloader': 4,
               'XnFlashFileTypeWavelengthCorrection': 26,
               'XnFlashFileTypeSensorTECParams': 21,
               'XnFlashFileTypeFileTable': 0,
               'XnFlashFileTypeDefaultParams': 9,
               'XnFlashFileTypeReferenceQVGA': 13,
               'XnFlashFileTypeGMCReferenceOffset': 27,
               'XnFlashFileTypeApplication': 6,
               'XnFlashFileTypeSensorAPCParams': 22,
               'XnFlashFileTypeBootSector': 2,
               'XnFlashFileTypeSensorFault': 29,
               'XnFlashFileTypeMonitor': 5,
               'XnFlashFileTypeDescriptors': 8,
               'XnFlashFileTypeImageCmos': 10,
               'XnFlashFileTypeProductionFile': 24,
               'XnFlashFileTypeScratchFile': 1,
               'XnFlashFileTypeReferenceVGA': 14,
               'XnFlashFileTypeMaintenance': 15,
               'XnFlashFileTypeSensorProjectorFaultParams': 23,
               'XnFlashFileTypeIDParams': 20,
               'XnFlashFileTypeUpgradeInProgress': 25,
               'XnFlashFileTypePrimeProcessor': 17,
               'XnFlashFileTypeDepthCmos': 11,
               'XnFlashFileTypeGainControl': 18,
               'XnFlashFileTypeVendorData': 30,
               'XnFlashFileTypeRegistartionParams': 19,
               'XnFlashFileTypeDebugParams': 16,
               'XnFlashFileTypeFixedParams': 7,
               'XnFlashFileTypeBootManager': 3,
               'XnFlashFileTypeSensorNESAParams': 28}
    _values_ = {0: 'XnFlashFileTypeFileTable',
                1: 'XnFlashFileTypeScratchFile',
                2: 'XnFlashFileTypeBootSector',
                3: 'XnFlashFileTypeBootManager',
                4: 'XnFlashFileTypeCodeDownloader',
                5: 'XnFlashFileTypeMonitor',
                6: 'XnFlashFileTypeApplication',
                7: 'XnFlashFileTypeFixedParams',
                8: 'XnFlashFileTypeDescriptors',
                9: 'XnFlashFileTypeDefaultParams',
                10: 'XnFlashFileTypeImageCmos',
                11: 'XnFlashFileTypeDepthCmos',
                12: 'XnFlashFileTypeAlgorithmParams',
                13: 'XnFlashFileTypeReferenceQVGA',
                14: 'XnFlashFileTypeReferenceVGA',
                15: 'XnFlashFileTypeMaintenance',
                16: 'XnFlashFileTypeDebugParams',
                17: 'XnFlashFileTypePrimeProcessor',
                18: 'XnFlashFileTypeGainControl',
                19: 'XnFlashFileTypeRegistartionParams',
                20: 'XnFlashFileTypeIDParams',
                21: 'XnFlashFileTypeSensorTECParams',
                22: 'XnFlashFileTypeSensorAPCParams',
                23: 'XnFlashFileTypeSensorProjectorFaultParams',
                24: 'XnFlashFileTypeProductionFile',
                25: 'XnFlashFileTypeUpgradeInProgress',
                26: 'XnFlashFileTypeWavelengthCorrection',
                27: 'XnFlashFileTypeGMCReferenceOffset',
                28: 'XnFlashFileTypeSensorNESAParams',
                29: 'XnFlashFileTypeSensorFault',
                30: 'XnFlashFileTypeVendorData'}
    XnFlashFileTypeFileTable = 0
    XnFlashFileTypeScratchFile = 1
    XnFlashFileTypeBootSector = 2
    XnFlashFileTypeBootManager = 3
    XnFlashFileTypeCodeDownloader = 4
    XnFlashFileTypeMonitor = 5
    XnFlashFileTypeApplication = 6
    XnFlashFileTypeFixedParams = 7
    XnFlashFileTypeDescriptors = 8
    XnFlashFileTypeDefaultParams = 9
    XnFlashFileTypeImageCmos = 10
    XnFlashFileTypeDepthCmos = 11
    XnFlashFileTypeAlgorithmParams = 12
    XnFlashFileTypeReferenceQVGA = 13
    XnFlashFileTypeReferenceVGA = 14
    XnFlashFileTypeMaintenance = 15
    XnFlashFileTypeDebugParams = 16
    XnFlashFileTypePrimeProcessor = 17
    XnFlashFileTypeGainControl = 18
    XnFlashFileTypeRegistartionParams = 19
    XnFlashFileTypeIDParams = 20
    XnFlashFileTypeSensorTECParams = 21
    XnFlashFileTypeSensorAPCParams = 22
    XnFlashFileTypeSensorProjectorFaultParams = 23
    XnFlashFileTypeProductionFile = 24
    XnFlashFileTypeUpgradeInProgress = 25
    XnFlashFileTypeWavelengthCorrection = 26
    XnFlashFileTypeGMCReferenceOffset = 27
    XnFlashFileTypeSensorNESAParams = 28
    XnFlashFileTypeSensorFault = 29
    XnFlashFileTypeVendorData = 30


class XnBistType(CEnum):
    _names_ = {'XN_BIST_FULL_FLASH': 16,
               'XN_BIST_FLASH': 8,
               'XN_BIST_TEC_TEST_MASK': 64,
               'XN_BIST_NESA_UNLIMITED_TEST_MASK': 256,
               'XN_BIST_POTENTIOMETER': 4,
               'XN_BIST_PROJECTOR_TEST_MASK': 32,
               'XN_BIST_IMAGE_CMOS': 1,
               'XN_BIST_NESA_TEST_MASK': 128,
               'XN_BIST_ALL': '((4294967295 & (~XN_BIST_NESA_TEST_MASK)) & (~XN_BIST_NESA_UNLIMITED_TEST_MASK))',
               'XN_BIST_IR_CMOS': 2}
    _values_ = {32: 'XN_BIST_PROJECTOR_TEST_MASK',
                1: 'XN_BIST_IMAGE_CMOS',
                2: 'XN_BIST_IR_CMOS',
                4: 'XN_BIST_POTENTIOMETER',
                8: 'XN_BIST_FLASH',
                64: 'XN_BIST_TEC_TEST_MASK',
                128: 'XN_BIST_NESA_TEST_MASK',
                256: 'XN_BIST_NESA_UNLIMITED_TEST_MASK',
                16: 'XN_BIST_FULL_FLASH',
                '((4294967295 & (~XN_BIST_NESA_TEST_MASK)) & (~XN_BIST_NESA_UNLIMITED_TEST_MASK))': 'XN_BIST_ALL'}
    XN_BIST_IMAGE_CMOS = 1
    XN_BIST_IR_CMOS = 2
    XN_BIST_POTENTIOMETER = 4
    XN_BIST_FLASH = 8
    XN_BIST_FULL_FLASH = 16
    XN_BIST_PROJECTOR_TEST_MASK = 32
    XN_BIST_TEC_TEST_MASK = 64
    XN_BIST_NESA_TEST_MASK = 128
    XN_BIST_NESA_UNLIMITED_TEST_MASK = 256
    XN_BIST_ALL = ((4294967295 & (~XN_BIST_NESA_TEST_MASK)) & (~XN_BIST_NESA_UNLIMITED_TEST_MASK))


class XnBistError(CEnum):
    _names_ = {'XN_BIST_COLOR_CMOS_CONTROL_BUS_FAILURE': 128,
               'XN_TEC_TEST_HEATER_CROSSED': 1048576,
               'XN_BIST_PROJECTOR_TEST_LD_FAIL': 65536,
               'XN_BIST_COLOR_CMOS_RESET_FAILUE': 1024,
               'XN_BIST_FLASH_WRITE_LINE_FAILURE': 2048,
               'XN_BIST_IR_CMOS_DATA_BUS_FAILURE': 4,
               'XN_BIST_IR_CMOS_TRIGGER_FAILURE': 32,
               'XN_BIST_RAM_TEST_FAILURE': 1,
               'XN_BIST_PROJECTOR_TEST_FAILSAFE_HIGH_FAIL': 262144,
               'XN_BIST_POTENTIOMETER_CONTROL_BUS_FAILURE': 8192,
               'XN_BIST_AUDIO_TEST_FAILURE': 32768,
               'XN_BIST_IR_CMOS_STROBE_FAILURE': 64,
               'XN_BIST_PROJECTOR_TEST_FAILSAFE_LOW_FAIL': 524288,
               'XN_BIST_IR_CMOS_RESET_FAILUE': 16,
               'XN_BIST_IR_CMOS_CONTROL_BUS_FAILURE': 2,
               'XN_BIST_PROJECTOR_TEST_LD_FAILSAFE_TRIG_FAIL': 131072,
               'XN_BIST_COLOR_CMOS_BAD_VERSION': 512,
               'XN_BIST_FLASH_TEST_FAILURE': 4096,
               'XN_BIST_COLOR_CMOS_DATA_BUS_FAILURE': 256,
               'XN_TEC_TEST_TEC_CROSSED': 4194304,
               'XN_TEC_TEST_TEC_FAULT': 8388608,
               'XN_BIST_POTENTIOMETER_FAILURE': 16384,
               'XN_BIST_IR_CMOS_BAD_VERSION': 8,
               'XN_TEC_TEST_HEATER_DISCONNETED': 2097152}
    _values_ = {4096: 'XN_BIST_FLASH_TEST_FAILURE',
                1: 'XN_BIST_RAM_TEST_FAILURE',
                2: 'XN_BIST_IR_CMOS_CONTROL_BUS_FAILURE',
                4: 'XN_BIST_IR_CMOS_DATA_BUS_FAILURE',
                8192: 'XN_BIST_POTENTIOMETER_CONTROL_BUS_FAILURE',
                8: 'XN_BIST_IR_CMOS_BAD_VERSION',
                128: 'XN_BIST_COLOR_CMOS_CONTROL_BUS_FAILURE',
                256: 'XN_BIST_COLOR_CMOS_DATA_BUS_FAILURE',
                16: 'XN_BIST_IR_CMOS_RESET_FAILUE',
                8388608: 'XN_TEC_TEST_TEC_FAULT',
                512: 'XN_BIST_COLOR_CMOS_BAD_VERSION',
                4194304: 'XN_TEC_TEST_TEC_CROSSED',
                524288: 'XN_BIST_PROJECTOR_TEST_FAILSAFE_LOW_FAIL',
                32: 'XN_BIST_IR_CMOS_TRIGGER_FAILURE',
                131072: 'XN_BIST_PROJECTOR_TEST_LD_FAILSAFE_TRIG_FAIL',
                262144: 'XN_BIST_PROJECTOR_TEST_FAILSAFE_HIGH_FAIL',
                1024: 'XN_BIST_COLOR_CMOS_RESET_FAILUE',
                16384: 'XN_BIST_POTENTIOMETER_FAILURE',
                1048576: 'XN_TEC_TEST_HEATER_CROSSED',
                32768: 'XN_BIST_AUDIO_TEST_FAILURE',
                64: 'XN_BIST_IR_CMOS_STROBE_FAILURE',
                2048: 'XN_BIST_FLASH_WRITE_LINE_FAILURE',
                2097152: 'XN_TEC_TEST_HEATER_DISCONNETED',
                65536: 'XN_BIST_PROJECTOR_TEST_LD_FAIL'}
    XN_BIST_RAM_TEST_FAILURE = 1
    XN_BIST_IR_CMOS_CONTROL_BUS_FAILURE = 2
    XN_BIST_IR_CMOS_DATA_BUS_FAILURE = 4
    XN_BIST_IR_CMOS_BAD_VERSION = 8
    XN_BIST_IR_CMOS_RESET_FAILUE = 16
    XN_BIST_IR_CMOS_TRIGGER_FAILURE = 32
    XN_BIST_IR_CMOS_STROBE_FAILURE = 64
    XN_BIST_COLOR_CMOS_CONTROL_BUS_FAILURE = 128
    XN_BIST_COLOR_CMOS_DATA_BUS_FAILURE = 256
    XN_BIST_COLOR_CMOS_BAD_VERSION = 512
    XN_BIST_COLOR_CMOS_RESET_FAILUE = 1024
    XN_BIST_FLASH_WRITE_LINE_FAILURE = 2048
    XN_BIST_FLASH_TEST_FAILURE = 4096
    XN_BIST_POTENTIOMETER_CONTROL_BUS_FAILURE = 8192
    XN_BIST_POTENTIOMETER_FAILURE = 16384
    XN_BIST_AUDIO_TEST_FAILURE = 32768
    XN_BIST_PROJECTOR_TEST_LD_FAIL = 65536
    XN_BIST_PROJECTOR_TEST_LD_FAILSAFE_TRIG_FAIL = 131072
    XN_BIST_PROJECTOR_TEST_FAILSAFE_HIGH_FAIL = 262144
    XN_BIST_PROJECTOR_TEST_FAILSAFE_LOW_FAIL = 524288
    XN_TEC_TEST_HEATER_CROSSED = 1048576
    XN_TEC_TEST_HEATER_DISCONNETED = 2097152
    XN_TEC_TEST_TEC_CROSSED = 4194304
    XN_TEC_TEST_TEC_FAULT = 8388608


class XnDepthCMOSType(CEnum):
    _names_ = {'XN_DEPTH_CMOS_AR130': 2, 'XN_DEPTH_CMOS_MT9M001': 1, 'XN_DEPTH_CMOS_NONE': 0}
    _values_ = {0: 'XN_DEPTH_CMOS_NONE', 1: 'XN_DEPTH_CMOS_MT9M001', 2: 'XN_DEPTH_CMOS_AR130'}
    XN_DEPTH_CMOS_NONE = 0
    XN_DEPTH_CMOS_MT9M001 = 1
    XN_DEPTH_CMOS_AR130 = 2


class XnImageCMOSType(CEnum):
    _names_ = {'XN_IMAGE_CMOS_NONE': 0, 'XN_IMAGE_CMOS_MT9M114': 3,
               'XN_IMAGE_CMOS_MT9M112': 1, 'XN_IMAGE_CMOS_MT9D131': 2}
    _values_ = {0: 'XN_IMAGE_CMOS_NONE', 1: 'XN_IMAGE_CMOS_MT9M112',
                2: 'XN_IMAGE_CMOS_MT9D131', 3: 'XN_IMAGE_CMOS_MT9M114'}
    XN_IMAGE_CMOS_NONE = 0
    XN_IMAGE_CMOS_MT9M112 = 1
    XN_IMAGE_CMOS_MT9D131 = 2
    XN_IMAGE_CMOS_MT9M114 = 3


class XnSDKVersion(ctypes.Structure):
    _packed_ = 1
    nMajor = 'ctypes.c_ubyte'
    nMinor = 'ctypes.c_ubyte'
    nMaintenance = 'ctypes.c_ubyte'
    nBuild = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnSDKVersion(nMajor = %r, nMinor = %r, nMaintenance = %r, nBuild = %r)' % (self.nMajor, self.nMinor, self.nMaintenance, self.nBuild)


class XnVersions(ctypes.Structure):
    _packed_ = 1
    nMajor = 'ctypes.c_ubyte'
    nMinor = 'ctypes.c_ubyte'
    nBuild = 'ctypes.c_ushort'
    nChip = 'ctypes.c_uint'
    nFPGA = 'ctypes.c_ushort'
    nSystemVersion = 'ctypes.c_ushort'
    SDK = 'XnSDKVersion'
    HWVer = 'XnHWVer'
    FWVer = 'XnFWVer'
    SensorVer = 'XnSensorVer'
    ChipVer = 'XnChipVer'

    def __repr__(self):
        return 'XnVersions(nMajor = %r, nMinor = %r, nBuild = %r, nChip = %r, nFPGA = %r, nSystemVersion = %r, SDK = %r, HWVer = %r, FWVer = %r, SensorVer = %r, ChipVer = %r)' % (self.nMajor, self.nMinor, self.nBuild, self.nChip, self.nFPGA, self.nSystemVersion, self.SDK, self.HWVer, self.FWVer, self.SensorVer, self.ChipVer)


class XnInnerParamData(ctypes.Structure):
    _packed_ = 1
    nParam = 'ctypes.c_ushort'
    nValue = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnInnerParamData(nParam = %r, nValue = %r)' % (self.nParam, self.nValue)


class XnDepthAGCBin(ctypes.Structure):
    _packed_ = 1
    nBin = 'ctypes.c_ushort'
    nMin = 'ctypes.c_ushort'
    nMax = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnDepthAGCBin(nBin = %r, nMin = %r, nMax = %r)' % (self.nBin, self.nMin, self.nMax)


class XnControlProcessingData(ctypes.Structure):
    _packed_ = 1
    nRegister = 'ctypes.c_ushort'
    nValue = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnControlProcessingData(nRegister = %r, nValue = %r)' % (self.nRegister, self.nValue)


class XnAHBData(ctypes.Structure):
    _packed_ = 1
    nRegister = 'ctypes.c_uint'
    nValue = 'ctypes.c_uint'
    nMask = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnAHBData(nRegister = %r, nValue = %r, nMask = %r)' % (self.nRegister, self.nValue, self.nMask)


class XnPixelRegistration(ctypes.Structure):
    _packed_ = 1
    nDepthX = 'ctypes.c_uint'
    nDepthY = 'ctypes.c_uint'
    nDepthValue = 'ctypes.c_ushort'
    nImageXRes = 'ctypes.c_uint'
    nImageYRes = 'ctypes.c_uint'
    nImageX = 'ctypes.c_uint'
    nImageY = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnPixelRegistration(nDepthX = %r, nDepthY = %r, nDepthValue = %r, nImageXRes = %r, nImageYRes = %r, nImageX = %r, nImageY = %r)' % (self.nDepthX, self.nDepthY, self.nDepthValue, self.nImageXRes, self.nImageYRes, self.nImageX, self.nImageY)


class XnLedState(ctypes.Structure):
    _packed_ = 1
    nLedID = 'ctypes.c_ushort'
    nState = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnLedState(nLedID = %r, nState = %r)' % (self.nLedID, self.nState)


class XnCmosBlankingTime(ctypes.Structure):
    _packed_ = 1
    nCmosID = 'XnCMOSType'
    nTimeInMilliseconds = 'ctypes.c_float'
    nNumberOfFrames = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnCmosBlankingTime(nCmosID = %r, nTimeInMilliseconds = %r, nNumberOfFrames = %r)' % (self.nCmosID, self.nTimeInMilliseconds, self.nNumberOfFrames)


class XnCmosBlankingUnits(ctypes.Structure):
    _packed_ = 1
    nCmosID = 'XnCMOSType'
    nUnits = 'ctypes.c_ushort'
    nNumberOfFrames = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnCmosBlankingUnits(nCmosID = %r, nUnits = %r, nNumberOfFrames = %r)' % (self.nCmosID, self.nUnits, self.nNumberOfFrames)


class XnI2CWriteData(ctypes.Structure):
    _packed_ = 1
    nBus = 'ctypes.c_ushort'
    nSlaveAddress = 'ctypes.c_ushort'
    cpWriteBuffer = '(ctypes.c_ushort * 10)'
    nWriteSize = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnI2CWriteData(nBus = %r, nSlaveAddress = %r, cpWriteBuffer = %r, nWriteSize = %r)' % (self.nBus, self.nSlaveAddress, self.cpWriteBuffer, self.nWriteSize)


class XnI2CReadData(ctypes.Structure):
    _packed_ = 1
    nBus = 'ctypes.c_ushort'
    nSlaveAddress = 'ctypes.c_ushort'
    cpReadBuffer = '(ctypes.c_ushort * 10)'
    cpWriteBuffer = '(ctypes.c_ushort * 10)'
    nReadSize = 'ctypes.c_ushort'
    nWriteSize = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnI2CReadData(nBus = %r, nSlaveAddress = %r, cpReadBuffer = %r, cpWriteBuffer = %r, nReadSize = %r, nWriteSize = %r)' % (self.nBus, self.nSlaveAddress, self.cpReadBuffer, self.cpWriteBuffer, self.nReadSize, self.nWriteSize)


class XnTecData(ctypes.Structure):
    _packed_ = 1
    m_SetPointVoltage = 'ctypes.c_ushort'
    m_CompensationVoltage = 'ctypes.c_ushort'
    m_TecDutyCycle = 'ctypes.c_ushort'
    m_HeatMode = 'ctypes.c_ushort'
    m_ProportionalError = 'ctypes.c_int'
    m_IntegralError = 'ctypes.c_int'
    m_DerivativeError = 'ctypes.c_int'
    m_ScanMode = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnTecData(m_SetPointVoltage = %r, m_CompensationVoltage = %r, m_TecDutyCycle = %r, m_HeatMode = %r, m_ProportionalError = %r, m_IntegralError = %r, m_DerivativeError = %r, m_ScanMode = %r)' % (self.m_SetPointVoltage, self.m_CompensationVoltage, self.m_TecDutyCycle, self.m_HeatMode, self.m_ProportionalError, self.m_IntegralError, self.m_DerivativeError, self.m_ScanMode)


class XnTecFastConvergenceData(ctypes.Structure):
    _packed_ = 1
    m_SetPointTemperature = 'ctypes.c_short'
    m_MeasuredTemperature = 'ctypes.c_short'
    m_ProportionalError = 'ctypes.c_int'
    m_IntegralError = 'ctypes.c_int'
    m_DerivativeError = 'ctypes.c_int'
    m_ScanMode = 'ctypes.c_ushort'
    m_HeatMode = 'ctypes.c_ushort'
    m_TecDutyCycle = 'ctypes.c_ushort'
    m_TemperatureRange = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnTecFastConvergenceData(m_SetPointTemperature = %r, m_MeasuredTemperature = %r, m_ProportionalError = %r, m_IntegralError = %r, m_DerivativeError = %r, m_ScanMode = %r, m_HeatMode = %r, m_TecDutyCycle = %r, m_TemperatureRange = %r)' % (self.m_SetPointTemperature, self.m_MeasuredTemperature, self.m_ProportionalError, self.m_IntegralError, self.m_DerivativeError, self.m_ScanMode, self.m_HeatMode, self.m_TecDutyCycle, self.m_TemperatureRange)


class XnEmitterData(ctypes.Structure):
    _packed_ = 1
    m_State = 'ctypes.c_ushort'
    m_SetPointVoltage = 'ctypes.c_ushort'
    m_SetPointClocks = 'ctypes.c_ushort'
    m_PD_Reading = 'ctypes.c_ushort'
    m_EmitterSet = 'ctypes.c_ushort'
    m_EmitterSettingLogic = 'ctypes.c_ushort'
    m_LightMeasureLogic = 'ctypes.c_ushort'
    m_IsAPCEnabled = 'ctypes.c_ushort'
    m_EmitterSetStepSize = 'ctypes.c_ushort'
    m_ApcTolerance = 'ctypes.c_ushort'
    m_SubClocking = 'ctypes.c_ushort'
    m_Precision = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnEmitterData(m_State = %r, m_SetPointVoltage = %r, m_SetPointClocks = %r, m_PD_Reading = %r, m_EmitterSet = %r, m_EmitterSettingLogic = %r, m_LightMeasureLogic = %r, m_IsAPCEnabled = %r, m_EmitterSetStepSize = %r, m_ApcTolerance = %r, m_SubClocking = %r, m_Precision = %r)' % (self.m_State, self.m_SetPointVoltage, self.m_SetPointClocks, self.m_PD_Reading, self.m_EmitterSet, self.m_EmitterSettingLogic, self.m_LightMeasureLogic, self.m_IsAPCEnabled, self.m_EmitterSetStepSize, self.m_ApcTolerance, self.m_SubClocking, self.m_Precision)


class XnFileAttributes(ctypes.Structure):
    _packed_ = 1
    nId = 'ctypes.c_ushort'
    nAttribs = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnFileAttributes(nId = %r, nAttribs = %r)' % (self.nId, self.nAttribs)


class XnParamFileData(ctypes.Structure):
    _packed_ = 1
    nOffset = 'ctypes.c_uint'
    strFileName = 'ctypes.c_char_p'
    nAttributes = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnParamFileData(nOffset = %r, strFileName = %r, nAttributes = %r)' % (self.nOffset, self.strFileName, self.nAttributes)


class XnParamFlashData(ctypes.Structure):
    _packed_ = 1
    nOffset = 'ctypes.c_uint'
    nSize = 'ctypes.c_uint'
    pData = 'ctypes.POINTER(ctypes.c_ubyte)'

    def __repr__(self):
        return 'XnParamFlashData(nOffset = %r, nSize = %r, pData = %r)' % (self.nOffset, self.nSize, self.pData)


class XnFlashFile(ctypes.Structure):
    _packed_ = 1
    nId = 'ctypes.c_ushort'
    nType = 'ctypes.c_ushort'
    nVersion = 'ctypes.c_uint'
    nOffset = 'ctypes.c_uint'
    nSize = 'ctypes.c_uint'
    nCrc = 'ctypes.c_ushort'
    nAttributes = 'ctypes.c_ushort'
    nReserve = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnFlashFile(nId = %r, nType = %r, nVersion = %r, nOffset = %r, nSize = %r, nCrc = %r, nAttributes = %r, nReserve = %r)' % (self.nId, self.nType, self.nVersion, self.nOffset, self.nSize, self.nCrc, self.nAttributes, self.nReserve)


class XnFlashFileList(ctypes.Structure):
    _packed_ = 1
    pFiles = 'ctypes.POINTER(XnFlashFile)'
    nFiles = 'ctypes.c_ushort'

    def __repr__(self):
        return 'XnFlashFileList(pFiles = %r, nFiles = %r)' % (self.pFiles, self.nFiles)


class XnProjectorFaultData(ctypes.Structure):
    _packed_ = 1
    nMinThreshold = 'ctypes.c_ushort'
    nMaxThreshold = 'ctypes.c_ushort'
    bProjectorFaultEvent = 'ctypes.c_int'

    def __repr__(self):
        return 'XnProjectorFaultData(nMinThreshold = %r, nMaxThreshold = %r, bProjectorFaultEvent = %r)' % (self.nMinThreshold, self.nMaxThreshold, self.bProjectorFaultEvent)


class XnBist(ctypes.Structure):
    _packed_ = 1
    nTestsMask = 'ctypes.c_uint'
    nFailures = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnBist(nTestsMask = %r, nFailures = %r)' % (self.nTestsMask, self.nFailures)


class _anon_enum_39(CEnum):
    _names_ = {'PS_PROPERTY_USB_INTERFACE': 489156609, 'PS_PROPERTY_DUMP_DATA': 489095169}
    _values_ = {489095169: 'PS_PROPERTY_DUMP_DATA', 489156609: 'PS_PROPERTY_USB_INTERFACE'}
    PS_PROPERTY_DUMP_DATA = 489095169
    PS_PROPERTY_USB_INTERFACE = 489156609


PS_PROPERTY_DUMP_DATA = _anon_enum_39.PS_PROPERTY_DUMP_DATA
PS_PROPERTY_USB_INTERFACE = _anon_enum_39.PS_PROPERTY_USB_INTERFACE


class _anon_enum_40(CEnum):
    _names_ = {'PS_COMMAND_SET_LOG_MASK_STATE': 489152531,
               'PS_COMMAND_AHB_READ': 489152513,
               'PS_COMMAND_GET_FILE_LIST': 489152523,
               'PS_COMMAND_POWER_RESET': 489152518,
               'PS_COMMAND_DUMP_ENDPOINT': 489152525,
               'PS_COMMAND_AHB_WRITE': 489152514,
               'PS_COMMAND_I2C_READ': 489152515,
               'PS_COMMAND_FORMAT_ZONE': 489152524,
               'PS_COMMAND_STOP_LOG': 489152533,
               'PS_COMMAND_BEGIN_FIRMWARE_UPDATE': 489152519,
               'PS_COMMAND_I2C_WRITE': 489152516,
               'PS_COMMAND_DOWNLOAD_FILE': 489152522,
               'PS_COMMAND_START_LOG': 489152532,
               'PS_COMMAND_SOFT_RESET': 489152517,
               'PS_COMMAND_GET_BIST_LIST': 489152527,
               'PS_COMMAND_EXECUTE_BIST': 489152528,
               'PS_COMMAND_USB_TEST': 489152529,
               'PS_COMMAND_END_FIRMWARE_UPDATE': 489152520,
               'PS_COMMAND_GET_LOG_MASK_LIST': 489152530,
               'PS_COMMAND_UPLOAD_FILE': 489152521,
               'PS_COMMAND_GET_I2C_DEVICE_LIST': 489152526}
    _values_ = {489152513: 'PS_COMMAND_AHB_READ',
                489152514: 'PS_COMMAND_AHB_WRITE',
                489152515: 'PS_COMMAND_I2C_READ',
                489152516: 'PS_COMMAND_I2C_WRITE',
                489152517: 'PS_COMMAND_SOFT_RESET',
                489152518: 'PS_COMMAND_POWER_RESET',
                489152519: 'PS_COMMAND_BEGIN_FIRMWARE_UPDATE',
                489152520: 'PS_COMMAND_END_FIRMWARE_UPDATE',
                489152521: 'PS_COMMAND_UPLOAD_FILE',
                489152522: 'PS_COMMAND_DOWNLOAD_FILE',
                489152523: 'PS_COMMAND_GET_FILE_LIST',
                489152524: 'PS_COMMAND_FORMAT_ZONE',
                489152525: 'PS_COMMAND_DUMP_ENDPOINT',
                489152526: 'PS_COMMAND_GET_I2C_DEVICE_LIST',
                489152527: 'PS_COMMAND_GET_BIST_LIST',
                489152528: 'PS_COMMAND_EXECUTE_BIST',
                489152529: 'PS_COMMAND_USB_TEST',
                489152530: 'PS_COMMAND_GET_LOG_MASK_LIST',
                489152531: 'PS_COMMAND_SET_LOG_MASK_STATE',
                489152532: 'PS_COMMAND_START_LOG',
                489152533: 'PS_COMMAND_STOP_LOG'}
    PS_COMMAND_AHB_READ = 489152513
    PS_COMMAND_AHB_WRITE = 489152514
    PS_COMMAND_I2C_READ = 489152515
    PS_COMMAND_I2C_WRITE = 489152516
    PS_COMMAND_SOFT_RESET = 489152517
    PS_COMMAND_POWER_RESET = 489152518
    PS_COMMAND_BEGIN_FIRMWARE_UPDATE = 489152519
    PS_COMMAND_END_FIRMWARE_UPDATE = 489152520
    PS_COMMAND_UPLOAD_FILE = 489152521
    PS_COMMAND_DOWNLOAD_FILE = 489152522
    PS_COMMAND_GET_FILE_LIST = 489152523
    PS_COMMAND_FORMAT_ZONE = 489152524
    PS_COMMAND_DUMP_ENDPOINT = 489152525
    PS_COMMAND_GET_I2C_DEVICE_LIST = 489152526
    PS_COMMAND_GET_BIST_LIST = 489152527
    PS_COMMAND_EXECUTE_BIST = 489152528
    PS_COMMAND_USB_TEST = 489152529
    PS_COMMAND_GET_LOG_MASK_LIST = 489152530
    PS_COMMAND_SET_LOG_MASK_STATE = 489152531
    PS_COMMAND_START_LOG = 489152532
    PS_COMMAND_STOP_LOG = 489152533


PS_COMMAND_AHB_READ = _anon_enum_40.PS_COMMAND_AHB_READ
PS_COMMAND_AHB_WRITE = _anon_enum_40.PS_COMMAND_AHB_WRITE
PS_COMMAND_I2C_READ = _anon_enum_40.PS_COMMAND_I2C_READ
PS_COMMAND_I2C_WRITE = _anon_enum_40.PS_COMMAND_I2C_WRITE
PS_COMMAND_SOFT_RESET = _anon_enum_40.PS_COMMAND_SOFT_RESET
PS_COMMAND_POWER_RESET = _anon_enum_40.PS_COMMAND_POWER_RESET
PS_COMMAND_BEGIN_FIRMWARE_UPDATE = _anon_enum_40.PS_COMMAND_BEGIN_FIRMWARE_UPDATE
PS_COMMAND_END_FIRMWARE_UPDATE = _anon_enum_40.PS_COMMAND_END_FIRMWARE_UPDATE
PS_COMMAND_UPLOAD_FILE = _anon_enum_40.PS_COMMAND_UPLOAD_FILE
PS_COMMAND_DOWNLOAD_FILE = _anon_enum_40.PS_COMMAND_DOWNLOAD_FILE
PS_COMMAND_GET_FILE_LIST = _anon_enum_40.PS_COMMAND_GET_FILE_LIST
PS_COMMAND_FORMAT_ZONE = _anon_enum_40.PS_COMMAND_FORMAT_ZONE
PS_COMMAND_DUMP_ENDPOINT = _anon_enum_40.PS_COMMAND_DUMP_ENDPOINT
PS_COMMAND_GET_I2C_DEVICE_LIST = _anon_enum_40.PS_COMMAND_GET_I2C_DEVICE_LIST
PS_COMMAND_GET_BIST_LIST = _anon_enum_40.PS_COMMAND_GET_BIST_LIST
PS_COMMAND_EXECUTE_BIST = _anon_enum_40.PS_COMMAND_EXECUTE_BIST
PS_COMMAND_USB_TEST = _anon_enum_40.PS_COMMAND_USB_TEST
PS_COMMAND_GET_LOG_MASK_LIST = _anon_enum_40.PS_COMMAND_GET_LOG_MASK_LIST
PS_COMMAND_SET_LOG_MASK_STATE = _anon_enum_40.PS_COMMAND_SET_LOG_MASK_STATE
PS_COMMAND_START_LOG = _anon_enum_40.PS_COMMAND_START_LOG
PS_COMMAND_STOP_LOG = _anon_enum_40.PS_COMMAND_STOP_LOG


class XnUsbInterfaceType(CEnum):
    _names_ = {'PS_USB_INTERFACE_ISO_ENDPOINTS': 1,
               'PS_USB_INTERFACE_DONT_CARE': 0, 'PS_USB_INTERFACE_BULK_ENDPOINTS': 2}
    _values_ = {0: 'PS_USB_INTERFACE_DONT_CARE',
                1: 'PS_USB_INTERFACE_ISO_ENDPOINTS', 2: 'PS_USB_INTERFACE_BULK_ENDPOINTS'}
    PS_USB_INTERFACE_DONT_CARE = 0
    PS_USB_INTERFACE_ISO_ENDPOINTS = 1
    PS_USB_INTERFACE_BULK_ENDPOINTS = 2


class XnFwFileVersion(ctypes.Structure):
    _packed_ = 1
    major = 'ctypes.c_ubyte'
    minor = 'ctypes.c_ubyte'
    maintenance = 'ctypes.c_ubyte'
    build = 'ctypes.c_ubyte'

    def __repr__(self):
        return 'XnFwFileVersion(major = %r, minor = %r, maintenance = %r, build = %r)' % (self.major, self.minor, self.maintenance, self.build)


class XnFwFileFlags(CEnum):
    _names_ = {'XN_FILE_FLAG_BAD_CRC': 1}
    _values_ = {1: 'XN_FILE_FLAG_BAD_CRC'}
    XN_FILE_FLAG_BAD_CRC = 1


class XnFwFileEntry(ctypes.Structure):
    _packed_ = 1
    name = '(ctypes.c_char * 32)'
    version = 'XnFwFileVersion'
    address = 'ctypes.c_uint'
    size = 'ctypes.c_uint'
    crc = 'ctypes.c_ushort'
    zone = 'ctypes.c_ushort'
    flags = 'XnFwFileFlags'

    def __repr__(self):
        return 'XnFwFileEntry(name = %r, version = %r, address = %r, size = %r, crc = %r, zone = %r, flags = %r)' % (self.name, self.version, self.address, self.size, self.crc, self.zone, self.flags)


class XnI2CDeviceInfo(ctypes.Structure):
    _packed_ = 1
    id = 'ctypes.c_uint'
    name = '(ctypes.c_char * 32)'

    def __repr__(self):
        return 'XnI2CDeviceInfo(id = %r, name = %r)' % (self.id, self.name)


class XnBistInfo(ctypes.Structure):
    _packed_ = 1
    id = 'ctypes.c_uint'
    name = '(ctypes.c_char * 32)'

    def __repr__(self):
        return 'XnBistInfo(id = %r, name = %r)' % (self.id, self.name)


class XnFwLogMask(ctypes.Structure):
    _packed_ = 1
    id = 'ctypes.c_uint'
    name = '(ctypes.c_char * 32)'

    def __repr__(self):
        return 'XnFwLogMask(id = %r, name = %r)' % (self.id, self.name)


class XnUsbTestEndpointResult(ctypes.Structure):
    _packed_ = 1
    averageBytesPerSecond = 'ctypes.c_double'
    lostPackets = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnUsbTestEndpointResult(averageBytesPerSecond = %r, lostPackets = %r)' % (self.averageBytesPerSecond, self.lostPackets)


class XnCommandAHB(ctypes.Structure):
    _packed_ = 1
    address = 'ctypes.c_uint'
    offsetInBits = 'ctypes.c_uint'
    widthInBits = 'ctypes.c_uint'
    value = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnCommandAHB(address = %r, offsetInBits = %r, widthInBits = %r, value = %r)' % (self.address, self.offsetInBits, self.widthInBits, self.value)


class XnCommandI2C(ctypes.Structure):
    _packed_ = 1
    deviceID = 'ctypes.c_uint'
    addressSize = 'ctypes.c_uint'
    address = 'ctypes.c_uint'
    valueSize = 'ctypes.c_uint'
    mask = 'ctypes.c_uint'
    value = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnCommandI2C(deviceID = %r, addressSize = %r, address = %r, valueSize = %r, mask = %r, value = %r)' % (self.deviceID, self.addressSize, self.address, self.valueSize, self.mask, self.value)


class XnCommandUploadFile(ctypes.Structure):
    _packed_ = 1
    filePath = 'ctypes.c_char_p'
    uploadToFactory = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnCommandUploadFile(filePath = %r, uploadToFactory = %r)' % (self.filePath, self.uploadToFactory)


class XnCommandDownloadFile(ctypes.Structure):
    _packed_ = 1
    zone = 'ctypes.c_ushort'
    firmwareFileName = 'ctypes.c_char_p'
    targetPath = 'ctypes.c_char_p'

    def __repr__(self):
        return 'XnCommandDownloadFile(zone = %r, firmwareFileName = %r, targetPath = %r)' % (self.zone, self.firmwareFileName, self.targetPath)


class XnCommandGetFileList(ctypes.Structure):
    _packed_ = 1
    count = 'ctypes.c_uint'
    files = 'ctypes.POINTER(XnFwFileEntry)'

    def __repr__(self):
        return 'XnCommandGetFileList(count = %r, files = %r)' % (self.count, self.files)


class XnCommandFormatZone(ctypes.Structure):
    _packed_ = 1
    zone = 'ctypes.c_ubyte'

    def __repr__(self):
        return 'XnCommandFormatZone(zone = %r)' % (self.zone)


class XnCommandDumpEndpoint(ctypes.Structure):
    _packed_ = 1
    endpoint = 'ctypes.c_ubyte'
    enabled = 'ctypes.c_bool'

    def __repr__(self):
        return 'XnCommandDumpEndpoint(endpoint = %r, enabled = %r)' % (self.endpoint, self.enabled)


class XnCommandGetI2CDeviceList(ctypes.Structure):
    _packed_ = 1
    count = 'ctypes.c_uint'
    devices = 'ctypes.POINTER(XnI2CDeviceInfo)'

    def __repr__(self):
        return 'XnCommandGetI2CDeviceList(count = %r, devices = %r)' % (self.count, self.devices)


class XnCommandGetBistList(ctypes.Structure):
    _packed_ = 1
    count = 'ctypes.c_uint'
    tests = 'ctypes.POINTER(XnBistInfo)'

    def __repr__(self):
        return 'XnCommandGetBistList(count = %r, tests = %r)' % (self.count, self.tests)


class XnCommandExecuteBist(ctypes.Structure):
    _packed_ = 1
    id = 'ctypes.c_uint'
    errorCode = 'ctypes.c_uint'
    extraDataSize = 'ctypes.c_uint'
    extraData = 'ctypes.POINTER(ctypes.c_ubyte)'

    def __repr__(self):
        return 'XnCommandExecuteBist(id = %r, errorCode = %r, extraDataSize = %r, extraData = %r)' % (self.id, self.errorCode, self.extraDataSize, self.extraData)


class XnCommandUsbTest(ctypes.Structure):
    _packed_ = 1
    seconds = 'ctypes.c_uint'
    endpointCount = 'ctypes.c_uint'
    endpoints = 'ctypes.POINTER(XnUsbTestEndpointResult)'

    def __repr__(self):
        return 'XnCommandUsbTest(seconds = %r, endpointCount = %r, endpoints = %r)' % (self.seconds, self.endpointCount, self.endpoints)


class XnCommandGetLogMaskList(ctypes.Structure):
    _packed_ = 1
    count = 'ctypes.c_uint'
    masks = 'ctypes.POINTER(XnFwLogMask)'

    def __repr__(self):
        return 'XnCommandGetLogMaskList(count = %r, masks = %r)' % (self.count, self.masks)


class XnCommandSetLogMaskState(ctypes.Structure):
    _packed_ = 1
    mask = 'ctypes.c_uint'
    enabled = 'ctypes.c_bool'

    def __repr__(self):
        return 'XnCommandSetLogMaskState(mask = %r, enabled = %r)' % (self.mask, self.enabled)


class _anon_enum_41(CEnum):
    _names_ = {'LINK_COMMAND_STOP_FW_STREAM': 302051333,
               'LINK_COMMAND_CREATE_FW_STREAM': 302051330,
               'LINK_PROP_PIXEL_FORMAT': 301993985,
               'LINK_PROP_DEPTH_SCALE': 301989899,
               'LINK_PROP_BOOT_STATUS': 301989899,
               'LINK_PROP_VERSIONS_INFO_COUNT': 301989890,
               'LINK_PROP_PRESET_FILE': 301989898,
               'LINK_PROP_CONST_SHIFT': 301998083,
               'LINK_COMMAND_GET_FW_STREAM_LIST': 302051329,
               'LINK_PROP_EMITTER_DEPTH_CMOS_DISTANCE': 301998088,
               'LINK_COMMAND_DESTROY_FW_STREAM': 302051331,
               'LINK_PROP_MAX_SHIFT': 301998081,
               'LINK_COMMAND_START_FW_STREAM': 302051332,
               'LINK_PROP_EMITTER_ACTIVE': 301989896,
               'LINK_COMMAND_SET_FW_STREAM_VIDEO_MODE': 302051335,
               'LINK_PROP_PARAM_COEFF': 301998084,
               'LINK_PROP_ZERO_PLANE_PIXEL_SIZE': 301998086,
               'LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE': 302051336,
               'LINK_PROP_SHIFT_TO_DEPTH_TABLE': 301998089,
               'LINK_PROP_COMPRESSION': 301993986,
               'LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE_LIST': 302051334,
               'LINK_PROP_SHIFT_SCALE': 301998085,
               'LINK_PROP_VERSIONS_INFO': 301989891,
               'LINK_PROP_FW_VERSION': 301989889,
               'LINK_PROP_ZERO_PLANE_DISTANCE': 301998082,
               'LINK_PROP_ZERO_PLANE_OUTPUT_PIXEL_SIZE': 301998087,
               'LINK_PROP_DEPTH_TO_SHIFT_TABLE': 301998090}
    _values_ = {301989889: 'LINK_PROP_FW_VERSION',
                301998082: 'LINK_PROP_ZERO_PLANE_DISTANCE',
                301989891: 'LINK_PROP_VERSIONS_INFO',
                302051332: 'LINK_COMMAND_START_FW_STREAM',
                302051333: 'LINK_COMMAND_STOP_FW_STREAM',
                302051334: 'LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE_LIST',
                302051329: 'LINK_COMMAND_GET_FW_STREAM_LIST',
                301989896: 'LINK_PROP_EMITTER_ACTIVE',
                301998089: 'LINK_PROP_SHIFT_TO_DEPTH_TABLE',
                301989898: 'LINK_PROP_PRESET_FILE',
                301989899: 'LINK_PROP_DEPTH_SCALE',
                301989890: 'LINK_PROP_VERSIONS_INFO_COUNT',
                301998083: 'LINK_PROP_CONST_SHIFT',
                301998084: 'LINK_PROP_PARAM_COEFF',
                301998085: 'LINK_PROP_SHIFT_SCALE',
                301993985: 'LINK_PROP_PIXEL_FORMAT',
                301998086: 'LINK_PROP_ZERO_PLANE_PIXEL_SIZE',
                302051335: 'LINK_COMMAND_SET_FW_STREAM_VIDEO_MODE',
                302051336: 'LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE',
                301998090: 'LINK_PROP_DEPTH_TO_SHIFT_TABLE',
                301998081: 'LINK_PROP_MAX_SHIFT',
                302051330: 'LINK_COMMAND_CREATE_FW_STREAM',
                301993986: 'LINK_PROP_COMPRESSION',
                301998087: 'LINK_PROP_ZERO_PLANE_OUTPUT_PIXEL_SIZE',
                302051331: 'LINK_COMMAND_DESTROY_FW_STREAM',
                301998088: 'LINK_PROP_EMITTER_DEPTH_CMOS_DISTANCE'}
    LINK_PROP_FW_VERSION = 301989889
    LINK_PROP_VERSIONS_INFO_COUNT = 301989890
    LINK_PROP_VERSIONS_INFO = 301989891
    LINK_PROP_EMITTER_ACTIVE = 301989896
    LINK_PROP_PRESET_FILE = 301989898
    LINK_PROP_BOOT_STATUS = 301989899
    LINK_COMMAND_GET_FW_STREAM_LIST = 302051329
    LINK_COMMAND_CREATE_FW_STREAM = 302051330
    LINK_COMMAND_DESTROY_FW_STREAM = 302051331
    LINK_COMMAND_START_FW_STREAM = 302051332
    LINK_COMMAND_STOP_FW_STREAM = 302051333
    LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE_LIST = 302051334
    LINK_COMMAND_SET_FW_STREAM_VIDEO_MODE = 302051335
    LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE = 302051336
    LINK_PROP_PIXEL_FORMAT = 301993985
    LINK_PROP_COMPRESSION = 301993986
    LINK_PROP_DEPTH_SCALE = 301989899
    LINK_PROP_MAX_SHIFT = 301998081
    LINK_PROP_ZERO_PLANE_DISTANCE = 301998082
    LINK_PROP_CONST_SHIFT = 301998083
    LINK_PROP_PARAM_COEFF = 301998084
    LINK_PROP_SHIFT_SCALE = 301998085
    LINK_PROP_ZERO_PLANE_PIXEL_SIZE = 301998086
    LINK_PROP_ZERO_PLANE_OUTPUT_PIXEL_SIZE = 301998087
    LINK_PROP_EMITTER_DEPTH_CMOS_DISTANCE = 301998088
    LINK_PROP_SHIFT_TO_DEPTH_TABLE = 301998089
    LINK_PROP_DEPTH_TO_SHIFT_TABLE = 301998090


LINK_PROP_FW_VERSION = _anon_enum_41.LINK_PROP_FW_VERSION
LINK_PROP_VERSIONS_INFO_COUNT = _anon_enum_41.LINK_PROP_VERSIONS_INFO_COUNT
LINK_PROP_VERSIONS_INFO = _anon_enum_41.LINK_PROP_VERSIONS_INFO
LINK_PROP_EMITTER_ACTIVE = _anon_enum_41.LINK_PROP_EMITTER_ACTIVE
LINK_PROP_PRESET_FILE = _anon_enum_41.LINK_PROP_PRESET_FILE
LINK_PROP_BOOT_STATUS = _anon_enum_41.LINK_PROP_BOOT_STATUS
LINK_COMMAND_GET_FW_STREAM_LIST = _anon_enum_41.LINK_COMMAND_GET_FW_STREAM_LIST
LINK_COMMAND_CREATE_FW_STREAM = _anon_enum_41.LINK_COMMAND_CREATE_FW_STREAM
LINK_COMMAND_DESTROY_FW_STREAM = _anon_enum_41.LINK_COMMAND_DESTROY_FW_STREAM
LINK_COMMAND_START_FW_STREAM = _anon_enum_41.LINK_COMMAND_START_FW_STREAM
LINK_COMMAND_STOP_FW_STREAM = _anon_enum_41.LINK_COMMAND_STOP_FW_STREAM
LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE_LIST = _anon_enum_41.LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE_LIST
LINK_COMMAND_SET_FW_STREAM_VIDEO_MODE = _anon_enum_41.LINK_COMMAND_SET_FW_STREAM_VIDEO_MODE
LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE = _anon_enum_41.LINK_COMMAND_GET_FW_STREAM_VIDEO_MODE
LINK_PROP_PIXEL_FORMAT = _anon_enum_41.LINK_PROP_PIXEL_FORMAT
LINK_PROP_COMPRESSION = _anon_enum_41.LINK_PROP_COMPRESSION
LINK_PROP_DEPTH_SCALE = _anon_enum_41.LINK_PROP_DEPTH_SCALE
LINK_PROP_MAX_SHIFT = _anon_enum_41.LINK_PROP_MAX_SHIFT
LINK_PROP_ZERO_PLANE_DISTANCE = _anon_enum_41.LINK_PROP_ZERO_PLANE_DISTANCE
LINK_PROP_CONST_SHIFT = _anon_enum_41.LINK_PROP_CONST_SHIFT
LINK_PROP_PARAM_COEFF = _anon_enum_41.LINK_PROP_PARAM_COEFF
LINK_PROP_SHIFT_SCALE = _anon_enum_41.LINK_PROP_SHIFT_SCALE
LINK_PROP_ZERO_PLANE_PIXEL_SIZE = _anon_enum_41.LINK_PROP_ZERO_PLANE_PIXEL_SIZE
LINK_PROP_ZERO_PLANE_OUTPUT_PIXEL_SIZE = _anon_enum_41.LINK_PROP_ZERO_PLANE_OUTPUT_PIXEL_SIZE
LINK_PROP_EMITTER_DEPTH_CMOS_DISTANCE = _anon_enum_41.LINK_PROP_EMITTER_DEPTH_CMOS_DISTANCE
LINK_PROP_SHIFT_TO_DEPTH_TABLE = _anon_enum_41.LINK_PROP_SHIFT_TO_DEPTH_TABLE
LINK_PROP_DEPTH_TO_SHIFT_TABLE = _anon_enum_41.LINK_PROP_DEPTH_TO_SHIFT_TABLE


class XnFileZone(CEnum):
    _names_ = {'XN_ZONE_UPDATE': 1,
               'XN_ZONE_FACTORY': 0}
    _values_ = {0: 'XN_ZONE_FACTORY',
                1: 'XN_ZONE_UPDATE'}
    XN_ZONE_FACTORY = 0
    XN_ZONE_UPDATE = 1


class XnBootErrorCode(CEnum):
    _names_ = {'XN_BOOT_OK': 0,
               'XN_BOOT_FW_LOAD_FAILED': 3,
               'XN_BOOT_BAD_CRC': 1,
               'XN_BOOT_UPLOAD_IN_PROGRESS': 2}
    _values_ = {0: 'XN_BOOT_OK',
                1: 'XN_BOOT_BAD_CRC',
                2: 'XN_BOOT_UPLOAD_IN_PROGRESS',
                3: 'XN_BOOT_FW_LOAD_FAILED'}
    XN_BOOT_OK = 0
    XN_BOOT_BAD_CRC = 1
    XN_BOOT_UPLOAD_IN_PROGRESS = 2
    XN_BOOT_FW_LOAD_FAILED = 3


class XnFwStreamType(CEnum):
    _names_ = {'XN_FW_STREAM_TYPE_IR': 2,
               'XN_FW_STREAM_TYPE_COLOR': 1,
               'XN_FW_STREAM_TYPE_SHIFTS': 3,
               'XN_FW_STREAM_TYPE_LOG': 8,
               'XN_FW_STREAM_TYPE_AUDIO': 4,
               'XN_FW_STREAM_TYPE_DY': 5}
    _values_ = {1: 'XN_FW_STREAM_TYPE_COLOR',
                2: 'XN_FW_STREAM_TYPE_IR',
                3: 'XN_FW_STREAM_TYPE_SHIFTS',
                4: 'XN_FW_STREAM_TYPE_AUDIO',
                5: 'XN_FW_STREAM_TYPE_DY',
                8: 'XN_FW_STREAM_TYPE_LOG'}
    XN_FW_STREAM_TYPE_COLOR = 1
    XN_FW_STREAM_TYPE_IR = 2
    XN_FW_STREAM_TYPE_SHIFTS = 3
    XN_FW_STREAM_TYPE_AUDIO = 4
    XN_FW_STREAM_TYPE_DY = 5
    XN_FW_STREAM_TYPE_LOG = 8


class XnFwPixelFormat(CEnum):
    _names_ = {'XN_FW_PIXEL_FORMAT_SHIFTS_9_3': 1,
               'XN_FW_PIXEL_FORMAT_NONE': 0,
               'XN_FW_PIXEL_FORMAT_GRAYSCALE16': 2,
               'XN_FW_PIXEL_FORMAT_BAYER8': 4,
               'XN_FW_PIXEL_FORMAT_YUV422': 3}
    _values_ = {0: 'XN_FW_PIXEL_FORMAT_NONE',
                1: 'XN_FW_PIXEL_FORMAT_SHIFTS_9_3',
                2: 'XN_FW_PIXEL_FORMAT_GRAYSCALE16',
                3: 'XN_FW_PIXEL_FORMAT_YUV422',
                4: 'XN_FW_PIXEL_FORMAT_BAYER8'}
    XN_FW_PIXEL_FORMAT_NONE = 0
    XN_FW_PIXEL_FORMAT_SHIFTS_9_3 = 1
    XN_FW_PIXEL_FORMAT_GRAYSCALE16 = 2
    XN_FW_PIXEL_FORMAT_YUV422 = 3
    XN_FW_PIXEL_FORMAT_BAYER8 = 4


class XnFwCompressionType(CEnum):
    _names_ = {'XN_FW_COMPRESSION_8Z': 1,
               'XN_FW_COMPRESSION_6_BIT_PACKED': 4,
               'XN_FW_COMPRESSION_16Z': 2,
               'XN_FW_COMPRESSION_24Z': 3,
               'XN_FW_COMPRESSION_11_BIT_PACKED': 6,
               'XN_FW_COMPRESSION_12_BIT_PACKED': 7,
               'XN_FW_COMPRESSION_10_BIT_PACKED': 5,
               'XN_FW_COMPRESSION_NONE': 0}
    _values_ = {0: 'XN_FW_COMPRESSION_NONE',
                1: 'XN_FW_COMPRESSION_8Z',
                2: 'XN_FW_COMPRESSION_16Z',
                3: 'XN_FW_COMPRESSION_24Z',
                4: 'XN_FW_COMPRESSION_6_BIT_PACKED',
                5: 'XN_FW_COMPRESSION_10_BIT_PACKED',
                6: 'XN_FW_COMPRESSION_11_BIT_PACKED',
                7: 'XN_FW_COMPRESSION_12_BIT_PACKED'}
    XN_FW_COMPRESSION_NONE = 0
    XN_FW_COMPRESSION_8Z = 1
    XN_FW_COMPRESSION_16Z = 2
    XN_FW_COMPRESSION_24Z = 3
    XN_FW_COMPRESSION_6_BIT_PACKED = 4
    XN_FW_COMPRESSION_10_BIT_PACKED = 5
    XN_FW_COMPRESSION_11_BIT_PACKED = 6
    XN_FW_COMPRESSION_12_BIT_PACKED = 7


class XnDetailedVersion(ctypes.Structure):
    _packed_ = 1
    m_nMajor = 'ctypes.c_ubyte'
    m_nMinor = 'ctypes.c_ubyte'
    m_nMaintenance = 'ctypes.c_ushort'
    m_nBuild = 'ctypes.c_uint'
    m_strModifier = '(ctypes.c_char * 16)'

    def __repr__(self):
        return 'XnDetailedVersion(m_nMajor = %r, m_nMinor = %r, m_nMaintenance = %r, m_nBuild = %r, m_strModifier = %r)' % (self.m_nMajor, self.m_nMinor, self.m_nMaintenance, self.m_nBuild, self.m_strModifier)


class XnBootStatus(ctypes.Structure):
    _packed_ = 1
    zone = 'XnFileZone'
    errorCode = 'XnBootErrorCode'

    def __repr__(self):
        return 'XnBootStatus(zone = %r, errorCode = %r)' % (self.zone, self.errorCode)


class XnFwStreamInfo(ctypes.Structure):
    _packed_ = 1
    type = 'XnFwStreamType'
    creationInfo = '(ctypes.c_char * 80)'

    def __repr__(self):
        return 'XnFwStreamInfo(type = %r, creationInfo = %r)' % (self.type, self.creationInfo)


class XnFwStreamVideoMode(ctypes.Structure):
    _packed_ = 1
    m_nXRes = 'ctypes.c_uint'
    m_nYRes = 'ctypes.c_uint'
    m_nFPS = 'ctypes.c_uint'
    m_nPixelFormat = 'XnFwPixelFormat'
    m_nCompression = 'XnFwCompressionType'

    def __repr__(self):
        return 'XnFwStreamVideoMode(m_nXRes = %r, m_nYRes = %r, m_nFPS = %r, m_nPixelFormat = %r, m_nCompression = %r)' % (self.m_nXRes, self.m_nYRes, self.m_nFPS, self.m_nPixelFormat, self.m_nCompression)


class XnCommandGetFwStreamList(ctypes.Structure):
    _packed_ = 1
    count = 'ctypes.c_uint'
    streams = 'ctypes.POINTER(XnFwStreamInfo)'

    def __repr__(self):
        return 'XnCommandGetFwStreamList(count = %r, streams = %r)' % (self.count, self.streams)


class XnCommandCreateStream(ctypes.Structure):
    _packed_ = 1
    type = 'XnFwStreamType'
    creationInfo = 'ctypes.c_char_p'
    id = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnCommandCreateStream(type = %r, creationInfo = %r, id = %r)' % (self.type, self.creationInfo, self.id)


class XnCommandDestroyStream(ctypes.Structure):
    _packed_ = 1
    id = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnCommandDestroyStream(id = %r)' % (self.id)


class XnCommandStartStream(ctypes.Structure):
    _packed_ = 1
    id = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnCommandStartStream(id = %r)' % (self.id)


class XnCommandStopStream(ctypes.Structure):
    _packed_ = 1
    id = 'ctypes.c_uint'

    def __repr__(self):
        return 'XnCommandStopStream(id = %r)' % (self.id)


class XnCommandGetFwStreamVideoModeList(ctypes.Structure):
    _packed_ = 1
    streamId = 'ctypes.c_int'
    count = 'ctypes.c_uint'
    videoModes = 'ctypes.POINTER(XnFwStreamVideoMode)'

    def __repr__(self):
        return 'XnCommandGetFwStreamVideoModeList(streamId = %r, count = %r, videoModes = %r)' % (self.streamId, self.count, self.videoModes)


class XnCommandSetFwStreamVideoMode(ctypes.Structure):
    _packed_ = 1
    streamId = 'ctypes.c_int'
    videoMode = 'XnFwStreamVideoMode'

    def __repr__(self):
        return 'XnCommandSetFwStreamVideoMode(streamId = %r, videoMode = %r)' % (self.streamId, self.videoMode)


class XnCommandGetFwStreamVideoMode(ctypes.Structure):
    _packed_ = 1
    streamId = 'ctypes.c_int'
    videoMode = 'XnFwStreamVideoMode'

    def __repr__(self):
        return 'XnCommandGetFwStreamVideoMode(streamId = %r, videoMode = %r)' % (self.streamId, self.videoMode)


OniBool = ctypes.c_int
OniCallbackHandle = ctypes.POINTER(OniCallbackHandleImpl)
OniHardwareVersion = ctypes.c_int
OniDeviceHandle = ctypes.POINTER(_OniDevice)
OniStreamHandle = ctypes.POINTER(_OniStream)
OniRecorderHandle = ctypes.POINTER(_OniRecorder)
OniNewFrameCallback = _get_calling_conv(None, OniStreamHandle, ctypes.c_void_p)
OniGeneralCallback = _get_calling_conv(None, ctypes.c_void_p)
OniDeviceInfoCallback = _get_calling_conv(None, ctypes.POINTER(OniDeviceInfo), ctypes.c_void_p)
OniDeviceStateCallback = _get_calling_conv(None, ctypes.POINTER(OniDeviceInfo), OniDeviceState, ctypes.c_void_p)
OniFrameAllocBufferCallback = _get_calling_conv(ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
OniFrameFreeBufferCallback = _get_calling_conv(None, ctypes.c_void_p, ctypes.c_void_p)
OniDepthPixel = ctypes.c_ushort
OniGrayscale16Pixel = ctypes.c_ushort
OniGrayscale8Pixel = ctypes.c_ubyte

OniCallbackHandleImpl._fields_ = [
]

AXonLinkSerialNumber._fields_ = [
    ('serial', ctypes.c_byte*32),
]

CamIntrinsicParam._fields_ = [
    ('ResolutionX', ctypes.c_int),
    ('ResolutionY', ctypes.c_int),
    ('fx', ctypes.c_float),
    ('fy', ctypes.c_float),
    ('cx', ctypes.c_float),
    ('cy', ctypes.c_float),
    ('k1', ctypes.c_float),
    ('k2', ctypes.c_float),
    ('k3', ctypes.c_float),
    ('p1', ctypes.c_float),
    ('p2', ctypes.c_float),
    ('k4', ctypes.c_float),
    ('k5', ctypes.c_float),
    ('k6', ctypes.c_float),
]

CamExtrinsicParam._fields_ = [
    ('R_Param', ctypes.c_float*9),
    ('T_Param', ctypes.c_float*3),
]

AXonLinkCamParam._fields_ = [
    ('stExtParam', CamExtrinsicParam),
    ('astDepthParam', CamIntrinsicParam*10),
    ('astColorParam', CamIntrinsicParam*10),
]

AXonLinkFWVersion._fields_ = [
    ( 'm_nhwType', ctypes.c_uint),
    ( 'm_nMajor', ctypes.c_uint),
    ( 'm_nMinor', ctypes.c_uint),
    ( 'm_nmaintenance', ctypes.c_uint),
    ( 'm_nbuild', ctypes.c_uint),
    ( 'm_nyear', ctypes.c_ubyte),
    ( 'm_nmonth', ctypes.c_ubyte),
    ( 'm_nday', ctypes.c_ubyte),
    ( 'm_nhour', ctypes.c_ubyte),
    ( 'm_nmin', ctypes.c_ubyte),
    ( 'm_nsec', ctypes.c_ubyte),
    ( 'reserved', ctypes.c_ubyte*34),

]

AXonLinkSWVersion._fields_ = [
    ( 'major', ctypes.c_uint),
    ( 'minor', ctypes.c_uint),
    ( 'maintenance', ctypes.c_uint),
    ( 'build', ctypes.c_uint),
]

AxonLinkFirmWarePacketVersion._fields_ = [
    ('filename', ctypes.c_char*256),
    ('hwType', ctypes.c_char),
    ('swVersion', AXonLinkSWVersion),
]

I2cValue._fields_ = [
    ('regaddress', ctypes.c_short),
    ('i2cvalue',ctypes.c_short),

]
E2Reg._fields_ = [
    ('tpye', ctypes.c_short),
    ('length', ctypes.c_short),
    ('crc', ctypes.c_short),
    ('data', ctypes.c_char_p),

]
AXonDSPInterface._fields_ = [
    ('UNDISTORT', ctypes.c_uint, 1),
    ('MASK', ctypes.c_uint, 1),
    ('NR3', ctypes.c_uint, 1),
    ('NR2', ctypes.c_uint, 1),
    ('GAMMA', ctypes.c_uint, 1),
    ('FLYING', ctypes.c_uint, 1),
    ('FLYING_2', ctypes.c_uint, 1),
    ('R2Z', ctypes.c_uint, 1),
    ('reserved', ctypes.c_uint, 24),
]
AXonLinkSetExposureLevel._fields_ = [
    ('curLevel', ctypes.c_ubyte),
    ('write2E2flag', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte*2),

]
AXonLinkGetExposureLevel._fields_ = [
    ('custonID', ctypes.c_ubyte),
    ('maxLevel', ctypes.c_ubyte),
    ('curLevel', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),

]
AXonCropping._fields_ = [
    ('originX', ctypes.c_ushort),
    ('originY', ctypes.c_ushort),
    ('width', ctypes.c_ushort),
    ('height', ctypes.c_ushort),
    ('gx', ctypes.c_ushort),
    ('gy', ctypes.c_ushort),

]
AXonLinkReadE2OnType._fields_ = [
    ('type', ctypes.c_ushort),
    ('Length', ctypes.c_ushort),
    ('data', ctypes.c_ubyte*500),
]
AXonCalibration._fields_ = [
    ('enable', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte*3)

]
AXonMotionThreshold._fields_ = [
    ('thresHold', ctypes.c_ushort),
    ('count', ctypes.c_uint),
    ('remain', ctypes.c_ushort)

]
AXonBoard_SN._fields_ = [
    ('len', ctypes.c_ushort),
    ('serialNumber', ctypes.c_char*30)

]
AXonSensorExposureWindow._fields_ = [
    ('cam_id', ctypes.c_ubyte),
    ('auto_mode', ctypes.c_ubyte),
    ('x', ctypes.c_ushort),
    ('y', ctypes.c_ushort),
    ('dx', ctypes.c_ushort),
    ('dy', ctypes.c_ushort),

]


OniVersion._fields_ = [
    ('major', ctypes.c_int),
    ('minor', ctypes.c_int),
    ('maintenance', ctypes.c_int),
    ('build', ctypes.c_int),
]

OniVideoMode._fields_ = [
    ('pixelFormat', OniPixelFormat),
    ('resolutionX', ctypes.c_int),
    ('resolutionY', ctypes.c_int),
    ('fps', ctypes.c_int),
]

OniSensorInfo._fields_ = [
    ('sensorType', OniSensorType),
    ('numSupportedVideoModes', ctypes.c_int),
    ('pSupportedVideoModes', ctypes.POINTER(OniVideoMode)),
]

OniDeviceInfo._fields_ = [
    ('uri', (ctypes.c_char * 256)),
    ('vendor', (ctypes.c_char * 256)),
    ('name', (ctypes.c_char * 256)),
    ('usbVendorId', ctypes.c_ushort),
    ('usbProductId', ctypes.c_ushort),
]

_OniDevice._fields_ = [
]

_OniStream._fields_ = [
]

_OniRecorder._fields_ = [
]

OniFrame._fields_ = [
    ('dataSize', ctypes.c_int),
    ('data', ctypes.c_void_p),
    ('sensorType', OniSensorType),
    ('timestamp', ctypes.c_ulonglong),
    ('frameIndex', ctypes.c_int),
    ('width', ctypes.c_int),
    ('height', ctypes.c_int),
    ('videoMode', OniVideoMode),
    ('croppingEnabled', OniBool),
    ('cropOriginX', ctypes.c_int),
    ('cropOriginY', ctypes.c_int),
    ('stride', ctypes.c_int),
]

OniDeviceCallbacks._fields_ = [
    ('deviceConnected', OniDeviceInfoCallback),
    ('deviceDisconnected', OniDeviceInfoCallback),
    ('deviceStateChanged', OniDeviceStateCallback),
]

OniCropping._fields_ = [
    ('enabled', ctypes.c_int),
    ('originX', ctypes.c_int),
    ('originY', ctypes.c_int),
    ('width', ctypes.c_int),
    ('height', ctypes.c_int),
]

OniRGB888Pixel._fields_ = [
    ('r', ctypes.c_ubyte),
    ('g', ctypes.c_ubyte),
    ('b', ctypes.c_ubyte),
]

OniYUV422DoublePixel._fields_ = [
    ('u', ctypes.c_ubyte),
    ('y1', ctypes.c_ubyte),
    ('v', ctypes.c_ubyte),
    ('y2', ctypes.c_ubyte),
]

OniSeek._fields_ = [
    ('frameIndex', ctypes.c_int),
    ('stream', OniStreamHandle),
]

XnSDKVersion._fields_ = [
    ('nMajor', ctypes.c_ubyte),
    ('nMinor', ctypes.c_ubyte),
    ('nMaintenance', ctypes.c_ubyte),
    ('nBuild', ctypes.c_ushort),
]

XnVersions._fields_ = [
    ('nMajor', ctypes.c_ubyte),
    ('nMinor', ctypes.c_ubyte),
    ('nBuild', ctypes.c_ushort),
    ('nChip', ctypes.c_uint),
    ('nFPGA', ctypes.c_ushort),
    ('nSystemVersion', ctypes.c_ushort),
    ('SDK', XnSDKVersion),
    ('HWVer', XnHWVer),
    ('FWVer', XnFWVer),
    ('SensorVer', XnSensorVer),
    ('ChipVer', XnChipVer),
]

XnInnerParamData._fields_ = [
    ('nParam', ctypes.c_ushort),
    ('nValue', ctypes.c_ushort),
]

XnDepthAGCBin._fields_ = [
    ('nBin', ctypes.c_ushort),
    ('nMin', ctypes.c_ushort),
    ('nMax', ctypes.c_ushort),
]

XnControlProcessingData._fields_ = [
    ('nRegister', ctypes.c_ushort),
    ('nValue', ctypes.c_ushort),
]

XnAHBData._fields_ = [
    ('nRegister', ctypes.c_uint),
    ('nValue', ctypes.c_uint),
    ('nMask', ctypes.c_uint),
]

XnPixelRegistration._fields_ = [
    ('nDepthX', ctypes.c_uint),
    ('nDepthY', ctypes.c_uint),
    ('nDepthValue', ctypes.c_ushort),
    ('nImageXRes', ctypes.c_uint),
    ('nImageYRes', ctypes.c_uint),
    ('nImageX', ctypes.c_uint),
    ('nImageY', ctypes.c_uint),
]

XnLedState._fields_ = [
    ('nLedID', ctypes.c_ushort),
    ('nState', ctypes.c_ushort),
]

XnCmosBlankingTime._fields_ = [
    ('nCmosID', XnCMOSType),
    ('nTimeInMilliseconds', ctypes.c_float),
    ('nNumberOfFrames', ctypes.c_ushort),
]

XnCmosBlankingUnits._fields_ = [
    ('nCmosID', XnCMOSType),
    ('nUnits', ctypes.c_ushort),
    ('nNumberOfFrames', ctypes.c_ushort),
]

XnI2CWriteData._fields_ = [
    ('nBus', ctypes.c_ushort),
    ('nSlaveAddress', ctypes.c_ushort),
    ('cpWriteBuffer', (ctypes.c_ushort * 10)),
    ('nWriteSize', ctypes.c_ushort),
]

XnI2CReadData._fields_ = [
    ('nBus', ctypes.c_ushort),
    ('nSlaveAddress', ctypes.c_ushort),
    ('cpReadBuffer', (ctypes.c_ushort * 10)),
    ('cpWriteBuffer', (ctypes.c_ushort * 10)),
    ('nReadSize', ctypes.c_ushort),
    ('nWriteSize', ctypes.c_ushort),
]

XnTecData._fields_ = [
    ('m_SetPointVoltage', ctypes.c_ushort),
    ('m_CompensationVoltage', ctypes.c_ushort),
    ('m_TecDutyCycle', ctypes.c_ushort),
    ('m_HeatMode', ctypes.c_ushort),
    ('m_ProportionalError', ctypes.c_int),
    ('m_IntegralError', ctypes.c_int),
    ('m_DerivativeError', ctypes.c_int),
    ('m_ScanMode', ctypes.c_ushort),
]

XnTecFastConvergenceData._fields_ = [
    ('m_SetPointTemperature', ctypes.c_short),
    ('m_MeasuredTemperature', ctypes.c_short),
    ('m_ProportionalError', ctypes.c_int),
    ('m_IntegralError', ctypes.c_int),
    ('m_DerivativeError', ctypes.c_int),
    ('m_ScanMode', ctypes.c_ushort),
    ('m_HeatMode', ctypes.c_ushort),
    ('m_TecDutyCycle', ctypes.c_ushort),
    ('m_TemperatureRange', ctypes.c_ushort),
]

XnEmitterData._fields_ = [
    ('m_State', ctypes.c_ushort),
    ('m_SetPointVoltage', ctypes.c_ushort),
    ('m_SetPointClocks', ctypes.c_ushort),
    ('m_PD_Reading', ctypes.c_ushort),
    ('m_EmitterSet', ctypes.c_ushort),
    ('m_EmitterSettingLogic', ctypes.c_ushort),
    ('m_LightMeasureLogic', ctypes.c_ushort),
    ('m_IsAPCEnabled', ctypes.c_ushort),
    ('m_EmitterSetStepSize', ctypes.c_ushort),
    ('m_ApcTolerance', ctypes.c_ushort),
    ('m_SubClocking', ctypes.c_ushort),
    ('m_Precision', ctypes.c_ushort),
]

XnFileAttributes._fields_ = [
    ('nId', ctypes.c_ushort),
    ('nAttribs', ctypes.c_ushort),
]

XnParamFileData._fields_ = [
    ('nOffset', ctypes.c_uint),
    ('strFileName', ctypes.c_char_p),
    ('nAttributes', ctypes.c_ushort),
]

XnParamFlashData._fields_ = [
    ('nOffset', ctypes.c_uint),
    ('nSize', ctypes.c_uint),
    ('pData', ctypes.POINTER(ctypes.c_ubyte)),
]

XnFlashFile._fields_ = [
    ('nId', ctypes.c_ushort),
    ('nType', ctypes.c_ushort),
    ('nVersion', ctypes.c_uint),
    ('nOffset', ctypes.c_uint),
    ('nSize', ctypes.c_uint),
    ('nCrc', ctypes.c_ushort),
    ('nAttributes', ctypes.c_ushort),
    ('nReserve', ctypes.c_ushort),
]

XnFlashFileList._fields_ = [
    ('pFiles', ctypes.POINTER(XnFlashFile)),
    ('nFiles', ctypes.c_ushort),
]

XnProjectorFaultData._fields_ = [
    ('nMinThreshold', ctypes.c_ushort),
    ('nMaxThreshold', ctypes.c_ushort),
    ('bProjectorFaultEvent', ctypes.c_int),
]

XnBist._fields_ = [
    ('nTestsMask', ctypes.c_uint),
    ('nFailures', ctypes.c_uint),
]

XnFwFileVersion._fields_ = [
    ('major', ctypes.c_ubyte),
    ('minor', ctypes.c_ubyte),
    ('maintenance', ctypes.c_ubyte),
    ('build', ctypes.c_ubyte),
]

XnFwFileEntry._fields_ = [
    ('name', (ctypes.c_char * 32)),
    ('version', XnFwFileVersion),
    ('address', ctypes.c_uint),
    ('size', ctypes.c_uint),
    ('crc', ctypes.c_ushort),
    ('zone', ctypes.c_ushort),
    ('flags', XnFwFileFlags),
]

XnI2CDeviceInfo._fields_ = [
    ('id', ctypes.c_uint),
    ('name', (ctypes.c_char * 32)),
]

XnBistInfo._fields_ = [
    ('id', ctypes.c_uint),
    ('name', (ctypes.c_char * 32)),
]

XnFwLogMask._fields_ = [
    ('id', ctypes.c_uint),
    ('name', (ctypes.c_char * 32)),
]

XnUsbTestEndpointResult._fields_ = [
    ('averageBytesPerSecond', ctypes.c_double),
    ('lostPackets', ctypes.c_uint),
]

XnCommandAHB._fields_ = [
    ('address', ctypes.c_uint),
    ('offsetInBits', ctypes.c_uint),
    ('widthInBits', ctypes.c_uint),
    ('value', ctypes.c_uint),
]

XnCommandI2C._fields_ = [
    ('deviceID', ctypes.c_uint),
    ('addressSize', ctypes.c_uint),
    ('address', ctypes.c_uint),
    ('valueSize', ctypes.c_uint),
    ('mask', ctypes.c_uint),
    ('value', ctypes.c_uint),
]

XnCommandUploadFile._fields_ = [
    ('filePath', ctypes.c_char_p),
    ('uploadToFactory', ctypes.c_uint),
]

XnCommandDownloadFile._fields_ = [
    ('zone', ctypes.c_ushort),
    ('firmwareFileName', ctypes.c_char_p),
    ('targetPath', ctypes.c_char_p),
]

XnCommandGetFileList._fields_ = [
    ('count', ctypes.c_uint),
    ('files', ctypes.POINTER(XnFwFileEntry)),
]

XnCommandFormatZone._fields_ = [
    ('zone', ctypes.c_ubyte),
]

XnCommandDumpEndpoint._fields_ = [
    ('endpoint', ctypes.c_ubyte),
    ('enabled', ctypes.c_bool),
]

XnCommandGetI2CDeviceList._fields_ = [
    ('count', ctypes.c_uint),
    ('devices', ctypes.POINTER(XnI2CDeviceInfo)),
]

XnCommandGetBistList._fields_ = [
    ('count', ctypes.c_uint),
    ('tests', ctypes.POINTER(XnBistInfo)),
]

XnCommandExecuteBist._fields_ = [
    ('id', ctypes.c_uint),
    ('errorCode', ctypes.c_uint),
    ('extraDataSize', ctypes.c_uint),
    ('extraData', ctypes.POINTER(ctypes.c_ubyte)),
]

XnCommandUsbTest._fields_ = [
    ('seconds', ctypes.c_uint),
    ('endpointCount', ctypes.c_uint),
    ('endpoints', ctypes.POINTER(XnUsbTestEndpointResult)),
]

XnCommandGetLogMaskList._fields_ = [
    ('count', ctypes.c_uint),
    ('masks', ctypes.POINTER(XnFwLogMask)),
]

XnCommandSetLogMaskState._fields_ = [
    ('mask', ctypes.c_uint),
    ('enabled', ctypes.c_bool),
]

XnDetailedVersion._fields_ = [
    ('m_nMajor', ctypes.c_ubyte),
    ('m_nMinor', ctypes.c_ubyte),
    ('m_nMaintenance', ctypes.c_ushort),
    ('m_nBuild', ctypes.c_uint),
    ('m_strModifier', (ctypes.c_char * 16)),
]

XnBootStatus._fields_ = [
    ('zone', XnFileZone),
    ('errorCode', XnBootErrorCode),
]

XnFwStreamInfo._fields_ = [
    ('type', XnFwStreamType),
    ('creationInfo', (ctypes.c_char * 80)),
]

XnFwStreamVideoMode._fields_ = [
    ('m_nXRes', ctypes.c_uint),
    ('m_nYRes', ctypes.c_uint),
    ('m_nFPS', ctypes.c_uint),
    ('m_nPixelFormat', XnFwPixelFormat),
    ('m_nCompression', XnFwCompressionType),
]

XnCommandGetFwStreamList._fields_ = [
    ('count', ctypes.c_uint),
    ('streams', ctypes.POINTER(XnFwStreamInfo)),
]

XnCommandCreateStream._fields_ = [
    ('type', XnFwStreamType),
    ('creationInfo', ctypes.c_char_p),
    ('id', ctypes.c_uint),
]

XnCommandDestroyStream._fields_ = [
    ('id', ctypes.c_uint),
]

XnCommandStartStream._fields_ = [
    ('id', ctypes.c_uint),
]

XnCommandStopStream._fields_ = [
    ('id', ctypes.c_uint),
]

XnCommandGetFwStreamVideoModeList._fields_ = [
    ('streamId', ctypes.c_int),
    ('count', ctypes.c_uint),
    ('videoModes', ctypes.POINTER(XnFwStreamVideoMode)),
]

XnCommandSetFwStreamVideoMode._fields_ = [
    ('streamId', ctypes.c_int),
    ('videoMode', XnFwStreamVideoMode),
]

XnCommandGetFwStreamVideoMode._fields_ = [
    ('streamId', ctypes.c_int),
    ('videoMode', XnFwStreamVideoMode),
]

_the_dll = UnloadedDLL
_oniInitialize = UnloadedDLL
_oniShutdown = UnloadedDLL
_oniGetDeviceList = UnloadedDLL
_oniReleaseDeviceList = UnloadedDLL
_oniRegisterDeviceCallbacks = UnloadedDLL
_oniUnregisterDeviceCallbacks = UnloadedDLL
_oniWaitForAnyStream = UnloadedDLL
_oniGetVersion = UnloadedDLL
_oniFormatBytesPerPixel = UnloadedDLL
_oniGetDepthValueUnit_mm = UnloadedDLL  # 添加获取深度值的单位
_oniGetExtendedError = UnloadedDLL
_oniDeviceOpen = UnloadedDLL
_oniDeviceClose = UnloadedDLL
_oniDeviceGetSensorInfo = UnloadedDLL
_oniDeviceGetInfo = UnloadedDLL
_oniDeviceCreateStream = UnloadedDLL
_oniDeviceEnableDepthColorSync = UnloadedDLL
_oniDeviceDisableDepthColorSync = UnloadedDLL
_oniDeviceGetDepthColorSyncEnabled = UnloadedDLL
_oniDeviceSetProperty = UnloadedDLL
_oniDeviceGetProperty = UnloadedDLL
_oniDeviceIsPropertySupported = UnloadedDLL
_oniDeviceInvoke = UnloadedDLL
_oniDeviceIsCommandSupported = UnloadedDLL
_oniDeviceIsImageRegistrationModeSupported = UnloadedDLL
_oniDeviceOpenEx = UnloadedDLL
_oniStreamDestroy = UnloadedDLL
_oniStreamGetSensorInfo = UnloadedDLL
_oniStreamStart = UnloadedDLL
_oniStreamStop = UnloadedDLL
_oniStreamReadFrame = UnloadedDLL
_oniStreamRegisterNewFrameCallback = UnloadedDLL
_oniStreamUnregisterNewFrameCallback = UnloadedDLL
_oniStreamSetProperty = UnloadedDLL
_oniStreamGetProperty = UnloadedDLL
_oniStreamIsPropertySupported = UnloadedDLL
_oniStreamInvoke = UnloadedDLL
_oniStreamIsCommandSupported = UnloadedDLL
_oniStreamSetFrameBuffersAllocator = UnloadedDLL
_oniFrameAddRef = UnloadedDLL
_oniFrameRelease = UnloadedDLL
_oniCreateRecorder = UnloadedDLL
_oniRecorderAttachStream = UnloadedDLL
_oniRecorderStart = UnloadedDLL
_oniRecorderStop = UnloadedDLL
_oniRecorderDestroy = UnloadedDLL
_oniCoordinateConverterDepthToWorld = UnloadedDLL
_oniCoordinateConverterWorldToDepth = UnloadedDLL
_oniCoordinateConverterDepthToColor = UnloadedDLL
_oniSetLogOutputFolder = UnloadedDLL
_oniGetLogFileName = UnloadedDLL
_oniSetLogMinSeverity = UnloadedDLL
_oniSetLogConsoleOutput = UnloadedDLL
_oniSetLogFileOutput = UnloadedDLL


def load_dll(dllname):
    global _the_dll
    if _the_dll:
        raise ValueError('DLL already loaded')
    dll = ctypes.CDLL(dllname)

    global _oniInitialize
    _oniInitialize = dll.oniInitialize
    _oniInitialize.restype = OniStatus
    _oniInitialize.argtypes = [ctypes.c_int]

    global _oniShutdown
    _oniShutdown = dll.oniShutdown
    _oniShutdown.restype = None
    _oniShutdown.argtypes = []

    global _oniGetDeviceList
    _oniGetDeviceList = dll.oniGetDeviceList
    _oniGetDeviceList.restype = OniStatus
    _oniGetDeviceList.argtypes = [ctypes.POINTER(ctypes.POINTER(OniDeviceInfo)), ctypes.POINTER(ctypes.c_int)]

    global _oniReleaseDeviceList
    _oniReleaseDeviceList = dll.oniReleaseDeviceList
    _oniReleaseDeviceList.restype = OniStatus
    _oniReleaseDeviceList.argtypes = [ctypes.POINTER(OniDeviceInfo)]

    global _oniRegisterDeviceCallbacks
    _oniRegisterDeviceCallbacks = dll.oniRegisterDeviceCallbacks
    _oniRegisterDeviceCallbacks.restype = OniStatus
    _oniRegisterDeviceCallbacks.argtypes = [ctypes.POINTER(
        OniDeviceCallbacks), ctypes.c_void_p, ctypes.POINTER(OniCallbackHandle)]

    global _oniUnregisterDeviceCallbacks
    _oniUnregisterDeviceCallbacks = dll.oniUnregisterDeviceCallbacks
    _oniUnregisterDeviceCallbacks.restype = None
    _oniUnregisterDeviceCallbacks.argtypes = [OniCallbackHandle]

    global _oniWaitForAnyStream
    _oniWaitForAnyStream = dll.oniWaitForAnyStream
    _oniWaitForAnyStream.restype = OniStatus
    _oniWaitForAnyStream.argtypes = [ctypes.POINTER(
        OniStreamHandle), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

    global _oniGetVersion
    _oniGetVersion = dll.oniGetVersion
    _oniGetVersion.restype = OniVersion
    _oniGetVersion.argtypes = []

    global _oniFormatBytesPerPixel
    _oniFormatBytesPerPixel = dll.oniFormatBytesPerPixel
    _oniFormatBytesPerPixel.restype = ctypes.c_int
    _oniFormatBytesPerPixel.argtypes = [OniPixelFormat]

    global _oniGetDepthValueUnit_mm
    _oniGetDepthValueUnit_mm = dll.oniGetDepthValueUnit_mm
    _oniGetDepthValueUnit_mm.restype = ctypes.c_float
    _oniGetDepthValueUnit_mm.argtypes = [OniPixelFormat]

    global _oniGetExtendedError
    _oniGetExtendedError = dll.oniGetExtendedError
    _oniGetExtendedError.restype = ctypes.c_char_p
    _oniGetExtendedError.argtypes = []

    global _oniDeviceOpen
    _oniDeviceOpen = dll.oniDeviceOpen
    _oniDeviceOpen.restype = OniStatus
    _oniDeviceOpen.argtypes = [ctypes.c_char_p, ctypes.POINTER(OniDeviceHandle)]

    global _oniDeviceClose
    _oniDeviceClose = dll.oniDeviceClose
    _oniDeviceClose.restype = OniStatus
    _oniDeviceClose.argtypes = [OniDeviceHandle]

    global _oniDeviceGetSensorInfo
    _oniDeviceGetSensorInfo = dll.oniDeviceGetSensorInfo
    _oniDeviceGetSensorInfo.restype = ctypes.POINTER(OniSensorInfo)
    _oniDeviceGetSensorInfo.argtypes = [OniDeviceHandle, OniSensorType]

    global _oniDeviceGetInfo
    _oniDeviceGetInfo = dll.oniDeviceGetInfo
    _oniDeviceGetInfo.restype = OniStatus
    _oniDeviceGetInfo.argtypes = [OniDeviceHandle, ctypes.POINTER(OniDeviceInfo)]

    global _oniDeviceCreateStream
    _oniDeviceCreateStream = dll.oniDeviceCreateStream
    _oniDeviceCreateStream.restype = OniStatus
    _oniDeviceCreateStream.argtypes = [OniDeviceHandle, OniSensorType, ctypes.POINTER(OniStreamHandle)]

    global _oniDeviceEnableDepthColorSync
    _oniDeviceEnableDepthColorSync = dll.oniDeviceEnableDepthColorSync
    _oniDeviceEnableDepthColorSync.restype = OniStatus
    _oniDeviceEnableDepthColorSync.argtypes = [OniDeviceHandle]

    global _oniDeviceDisableDepthColorSync
    _oniDeviceDisableDepthColorSync = dll.oniDeviceDisableDepthColorSync
    _oniDeviceDisableDepthColorSync.restype = None
    _oniDeviceDisableDepthColorSync.argtypes = [OniDeviceHandle]

    global _oniDeviceGetDepthColorSyncEnabled
    _oniDeviceGetDepthColorSyncEnabled = dll.oniDeviceGetDepthColorSyncEnabled
    _oniDeviceGetDepthColorSyncEnabled.restype = OniBool
    _oniDeviceGetDepthColorSyncEnabled.argtypes = [OniDeviceHandle]

    global _oniDeviceSetProperty
    _oniDeviceSetProperty = dll.oniDeviceSetProperty
    _oniDeviceSetProperty.restype = OniStatus
    _oniDeviceSetProperty.argtypes = [OniDeviceHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    global _oniDeviceGetProperty
    _oniDeviceGetProperty = dll.oniDeviceGetProperty
    _oniDeviceGetProperty.restype = OniStatus
    _oniDeviceGetProperty.argtypes = [OniDeviceHandle, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    global _oniDeviceIsPropertySupported
    _oniDeviceIsPropertySupported = dll.oniDeviceIsPropertySupported
    _oniDeviceIsPropertySupported.restype = OniBool
    _oniDeviceIsPropertySupported.argtypes = [OniDeviceHandle, ctypes.c_int]

    global _oniDeviceInvoke
    _oniDeviceInvoke = dll.oniDeviceInvoke
    _oniDeviceInvoke.restype = OniStatus
    _oniDeviceInvoke.argtypes = [OniDeviceHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    global _oniDeviceIsCommandSupported
    _oniDeviceIsCommandSupported = dll.oniDeviceIsCommandSupported
    _oniDeviceIsCommandSupported.restype = OniBool
    _oniDeviceIsCommandSupported.argtypes = [OniDeviceHandle, ctypes.c_int]

    global _oniDeviceIsImageRegistrationModeSupported
    _oniDeviceIsImageRegistrationModeSupported = dll.oniDeviceIsImageRegistrationModeSupported
    _oniDeviceIsImageRegistrationModeSupported.restype = OniBool
    _oniDeviceIsImageRegistrationModeSupported.argtypes = [OniDeviceHandle, OniImageRegistrationMode]

    global _oniDeviceOpenEx
    _oniDeviceOpenEx = dll.oniDeviceOpenEx
    _oniDeviceOpenEx.restype = OniStatus
    _oniDeviceOpenEx.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(OniDeviceHandle)]

    global _oniStreamDestroy
    _oniStreamDestroy = dll.oniStreamDestroy
    _oniStreamDestroy.restype = None
    _oniStreamDestroy.argtypes = [OniStreamHandle]

    global _oniStreamGetSensorInfo
    _oniStreamGetSensorInfo = dll.oniStreamGetSensorInfo
    _oniStreamGetSensorInfo.restype = ctypes.POINTER(OniSensorInfo)
    _oniStreamGetSensorInfo.argtypes = [OniStreamHandle]

    global _oniStreamStart
    _oniStreamStart = dll.oniStreamStart
    _oniStreamStart.restype = OniStatus
    _oniStreamStart.argtypes = [OniStreamHandle]

    global _oniStreamStop
    _oniStreamStop = dll.oniStreamStop
    _oniStreamStop.restype = None
    _oniStreamStop.argtypes = [OniStreamHandle]

    global _oniStreamReadFrame
    _oniStreamReadFrame = dll.oniStreamReadFrame
    _oniStreamReadFrame.restype = OniStatus
    _oniStreamReadFrame.argtypes = [OniStreamHandle, ctypes.POINTER(ctypes.POINTER(OniFrame))]

    global _oniStreamRegisterNewFrameCallback
    _oniStreamRegisterNewFrameCallback = dll.oniStreamRegisterNewFrameCallback
    _oniStreamRegisterNewFrameCallback.restype = OniStatus
    _oniStreamRegisterNewFrameCallback.argtypes = [OniStreamHandle,
                                                   OniNewFrameCallback, ctypes.c_void_p, ctypes.POINTER(OniCallbackHandle)]

    global _oniStreamUnregisterNewFrameCallback
    _oniStreamUnregisterNewFrameCallback = dll.oniStreamUnregisterNewFrameCallback
    _oniStreamUnregisterNewFrameCallback.restype = None
    _oniStreamUnregisterNewFrameCallback.argtypes = [OniStreamHandle, OniCallbackHandle]

    global _oniStreamSetProperty
    _oniStreamSetProperty = dll.oniStreamSetProperty
    _oniStreamSetProperty.restype = OniStatus
    _oniStreamSetProperty.argtypes = [OniStreamHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    global _oniStreamGetProperty
    _oniStreamGetProperty = dll.oniStreamGetProperty
    _oniStreamGetProperty.restype = OniStatus
    _oniStreamGetProperty.argtypes = [OniStreamHandle, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    global _oniStreamIsPropertySupported
    _oniStreamIsPropertySupported = dll.oniStreamIsPropertySupported
    _oniStreamIsPropertySupported.restype = OniBool
    _oniStreamIsPropertySupported.argtypes = [OniStreamHandle, ctypes.c_int]

    global _oniStreamInvoke
    _oniStreamInvoke = dll.oniStreamInvoke
    _oniStreamInvoke.restype = OniStatus
    _oniStreamInvoke.argtypes = [OniStreamHandle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    global _oniStreamIsCommandSupported
    _oniStreamIsCommandSupported = dll.oniStreamIsCommandSupported
    _oniStreamIsCommandSupported.restype = OniBool
    _oniStreamIsCommandSupported.argtypes = [OniStreamHandle, ctypes.c_int]

    global _oniStreamSetFrameBuffersAllocator
    _oniStreamSetFrameBuffersAllocator = dll.oniStreamSetFrameBuffersAllocator
    _oniStreamSetFrameBuffersAllocator.restype = OniStatus
    _oniStreamSetFrameBuffersAllocator.argtypes = [
        OniStreamHandle, OniFrameAllocBufferCallback, OniFrameFreeBufferCallback, ctypes.c_void_p]

    global _oniFrameAddRef
    _oniFrameAddRef = dll.oniFrameAddRef
    _oniFrameAddRef.restype = None
    _oniFrameAddRef.argtypes = [ctypes.POINTER(OniFrame)]

    global _oniFrameRelease
    _oniFrameRelease = dll.oniFrameRelease
    _oniFrameRelease.restype = None
    _oniFrameRelease.argtypes = [ctypes.POINTER(OniFrame)]

    global _oniCreateRecorder
    _oniCreateRecorder = dll.oniCreateRecorder
    _oniCreateRecorder.restype = OniStatus
    _oniCreateRecorder.argtypes = [ctypes.c_char_p, ctypes.POINTER(OniRecorderHandle)]

    global _oniRecorderAttachStream
    _oniRecorderAttachStream = dll.oniRecorderAttachStream
    _oniRecorderAttachStream.restype = OniStatus
    _oniRecorderAttachStream.argtypes = [OniRecorderHandle, OniStreamHandle, OniBool]

    global _oniRecorderStart
    _oniRecorderStart = dll.oniRecorderStart
    _oniRecorderStart.restype = OniStatus
    _oniRecorderStart.argtypes = [OniRecorderHandle]

    global _oniRecorderStop
    _oniRecorderStop = dll.oniRecorderStop
    _oniRecorderStop.restype = None
    _oniRecorderStop.argtypes = [OniRecorderHandle]

    global _oniRecorderDestroy
    _oniRecorderDestroy = dll.oniRecorderDestroy
    _oniRecorderDestroy.restype = OniStatus
    _oniRecorderDestroy.argtypes = [ctypes.POINTER(OniRecorderHandle)]

    global _oniCoordinateConverterDepthToWorld
    _oniCoordinateConverterDepthToWorld = dll.oniCoordinateConverterDepthToWorld
    _oniCoordinateConverterDepthToWorld.restype = OniStatus
    _oniCoordinateConverterDepthToWorld.argtypes = [OniStreamHandle, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(
        ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

    global _oniCoordinateConverterWorldToDepth
    _oniCoordinateConverterWorldToDepth = dll.oniCoordinateConverterWorldToDepth
    _oniCoordinateConverterWorldToDepth.restype = OniStatus
    _oniCoordinateConverterWorldToDepth.argtypes = [OniStreamHandle, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(
        ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

    global _oniCoordinateConverterDepthToColor
    _oniCoordinateConverterDepthToColor = dll.oniCoordinateConverterDepthToColor
    _oniCoordinateConverterDepthToColor.restype = OniStatus
    _oniCoordinateConverterDepthToColor.argtypes = [OniStreamHandle, OniStreamHandle, ctypes.c_int,
                                                    ctypes.c_int, OniDepthPixel, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

    global _oniSetLogOutputFolder
    _oniSetLogOutputFolder = dll.oniSetLogOutputFolder
    _oniSetLogOutputFolder.restype = OniStatus
    _oniSetLogOutputFolder.argtypes = [ctypes.c_char_p]

    global _oniGetLogFileName
    _oniGetLogFileName = dll.oniGetLogFileName
    _oniGetLogFileName.restype = OniStatus
    _oniGetLogFileName.argtypes = [ctypes.c_char_p, ctypes.c_int]

    global _oniSetLogMinSeverity
    _oniSetLogMinSeverity = dll.oniSetLogMinSeverity
    _oniSetLogMinSeverity.restype = OniStatus
    _oniSetLogMinSeverity.argtypes = [ctypes.c_int]

    global _oniSetLogConsoleOutput
    _oniSetLogConsoleOutput = dll.oniSetLogConsoleOutput
    _oniSetLogConsoleOutput.restype = OniStatus
    _oniSetLogConsoleOutput.argtypes = [OniBool]

    global _oniSetLogFileOutput
    _oniSetLogFileOutput = dll.oniSetLogFileOutput
    _oniSetLogFileOutput.restype = OniStatus
    _oniSetLogFileOutput.argtypes = [OniBool]

    _the_dll = dll


import functools
from axonopenni.utils import OpenNIError


def oni_call(func):
    @functools.wraps(func)
    def wrapper(*args):
        res = func(*args)
        if res != OniStatus.ONI_STATUS_OK:
            msg = oniGetExtendedError()
            if not msg:
                msg = ''
            buf = ctypes.create_string_buffer(1024)
            rc = _oniGetLogFileName(buf, ctypes.sizeof(buf))
            if rc == OniStatus.ONI_STATUS_OK:
                logfile = buf.value
            else:
                logfile = None
            raise OpenNIError(res, msg.strip(), logfile)
        return res

    return wrapper


@oni_call
def oniInitialize(apiVersion):
    '''OniStatus oniInitialize(int apiVersion)'''
    return _oniInitialize(apiVersion)


def oniShutdown():
    '''void oniShutdown()'''
    return _oniShutdown()


@oni_call
def oniGetDeviceList(pDevices, pNumDevices):
    '''OniStatus oniGetDeviceList(OniDeviceInfo** pDevices, int* pNumDevices)'''
    return _oniGetDeviceList(pDevices, pNumDevices)


@oni_call
def oniReleaseDeviceList(pDevices):
    '''OniStatus oniReleaseDeviceList(OniDeviceInfo* pDevices)'''
    return _oniReleaseDeviceList(pDevices)


@oni_call
def oniRegisterDeviceCallbacks(pCallbacks, pCookie, pHandle):
    '''OniStatus oniRegisterDeviceCallbacks(OniDeviceCallbacks* pCallbacks, void* pCookie, OniCallbackHandle* pHandle)'''
    return _oniRegisterDeviceCallbacks(pCallbacks, pCookie, pHandle)


def oniUnregisterDeviceCallbacks(handle):
    '''void oniUnregisterDeviceCallbacks(OniCallbackHandle handle)'''
    return _oniUnregisterDeviceCallbacks(handle)


@oni_call
def oniWaitForAnyStream(pStreams, numStreams, pStreamIndex, timeout):
    '''OniStatus oniWaitForAnyStream(OniStreamHandle* pStreams, int numStreams, int* pStreamIndex, int timeout)'''
    return _oniWaitForAnyStream(pStreams, numStreams, pStreamIndex, timeout)


def oniGetVersion():
    '''OniVersion oniGetVersion()'''
    return _oniGetVersion()


def oniFormatBytesPerPixel(format):
    '''int oniFormatBytesPerPixel(OniPixelFormat format)'''
    return _oniFormatBytesPerPixel(format)

def oniGetDepthValueUnit_mm(format):
    return _oniGetDepthValueUnit_mm(format)

def oniGetExtendedError():
    '''char* oniGetExtendedError()'''
    return _oniGetExtendedError()


@oni_call
def oniDeviceOpen(uri, pDevice):
    '''OniStatus oniDeviceOpen(char* uri, OniDeviceHandle* pDevice)'''
    return _oniDeviceOpen(uri, pDevice)


@oni_call
def oniDeviceClose(device):
    '''OniStatus oniDeviceClose(OniDeviceHandle device)'''
    return _oniDeviceClose(device)


def oniDeviceGetSensorInfo(device, sensorType):
    '''OniSensorInfo* oniDeviceGetSensorInfo(OniDeviceHandle device, OniSensorType sensorType)'''
    return _oniDeviceGetSensorInfo(device, sensorType)


@oni_call
def oniDeviceGetInfo(device, pInfo):
    '''OniStatus oniDeviceGetInfo(OniDeviceHandle device, OniDeviceInfo* pInfo)'''
    return _oniDeviceGetInfo(device, pInfo)


@oni_call
def oniDeviceCreateStream(device, sensorType, pStream):
    '''OniStatus oniDeviceCreateStream(OniDeviceHandle device, OniSensorType sensorType, OniStreamHandle* pStream)'''
    return _oniDeviceCreateStream(device, sensorType, pStream)


@oni_call
def oniDeviceEnableDepthColorSync(device):
    '''OniStatus oniDeviceEnableDepthColorSync(OniDeviceHandle device)'''
    return _oniDeviceEnableDepthColorSync(device)


def oniDeviceDisableDepthColorSync(device):
    '''void oniDeviceDisableDepthColorSync(OniDeviceHandle device)'''
    return _oniDeviceDisableDepthColorSync(device)


def oniDeviceGetDepthColorSyncEnabled(device):
    '''OniBool oniDeviceGetDepthColorSyncEnabled(OniDeviceHandle device)'''
    return _oniDeviceGetDepthColorSyncEnabled(device)


@oni_call
def oniDeviceSetProperty(device, propertyId, data, dataSize):
    '''OniStatus oniDeviceSetProperty(OniDeviceHandle device, int propertyId, void* data, int dataSize)'''
    return _oniDeviceSetProperty(device, propertyId, data, dataSize)


@oni_call
def oniDeviceGetProperty(device, propertyId, data, pDataSize):
    '''OniStatus oniDeviceGetProperty(OniDeviceHandle device, int propertyId, void* data, int* pDataSize)'''
    return _oniDeviceGetProperty(device, propertyId, data, pDataSize)


def oniDeviceIsPropertySupported(device, propertyId):
    '''OniBool oniDeviceIsPropertySupported(OniDeviceHandle device, int propertyId)'''
    return _oniDeviceIsPropertySupported(device, propertyId)


@oni_call
def oniDeviceInvoke(device, commandId, data, dataSize):
    '''OniStatus oniDeviceInvoke(OniDeviceHandle device, int commandId, void* data, int dataSize)'''
    return _oniDeviceInvoke(device, commandId, data, dataSize)


def oniDeviceIsCommandSupported(device, commandId):
    '''OniBool oniDeviceIsCommandSupported(OniDeviceHandle device, int commandId)'''
    return _oniDeviceIsCommandSupported(device, commandId)


def oniDeviceIsImageRegistrationModeSupported(device, mode):
    '''OniBool oniDeviceIsImageRegistrationModeSupported(OniDeviceHandle device, OniImageRegistrationMode mode)'''
    return _oniDeviceIsImageRegistrationModeSupported(device, mode)


@oni_call
def oniDeviceOpenEx(uri, mode, pDevice):
    '''OniStatus oniDeviceOpenEx(char* uri, char* mode, OniDeviceHandle* pDevice)'''
    return _oniDeviceOpenEx(uri, mode, pDevice)


def oniStreamDestroy(stream):
    '''void oniStreamDestroy(OniStreamHandle stream)'''
    return _oniStreamDestroy(stream)


def oniStreamGetSensorInfo(stream):
    '''OniSensorInfo* oniStreamGetSensorInfo(OniStreamHandle stream)'''
    return _oniStreamGetSensorInfo(stream)


@oni_call
def oniStreamStart(stream):
    '''OniStatus oniStreamStart(OniStreamHandle stream)'''
    return _oniStreamStart(stream)


def oniStreamStop(stream):
    '''void oniStreamStop(OniStreamHandle stream)'''
    return _oniStreamStop(stream)


@oni_call
def oniStreamReadFrame(stream, pFrame):
    '''OniStatus oniStreamReadFrame(OniStreamHandle stream, OniFrame** pFrame)'''
    return _oniStreamReadFrame(stream, pFrame)


@oni_call
def oniStreamRegisterNewFrameCallback(stream, handler, pCookie, pHandle):
    '''OniStatus oniStreamRegisterNewFrameCallback(OniStreamHandle stream, OniNewFrameCallback handler, void* pCookie, OniCallbackHandle* pHandle)'''
    return _oniStreamRegisterNewFrameCallback(stream, handler, pCookie, pHandle)


def oniStreamUnregisterNewFrameCallback(stream, handle):
    '''void oniStreamUnregisterNewFrameCallback(OniStreamHandle stream, OniCallbackHandle handle)'''
    return _oniStreamUnregisterNewFrameCallback(stream, handle)


@oni_call
def oniStreamSetProperty(stream, propertyId, data, dataSize):
    '''OniStatus oniStreamSetProperty(OniStreamHandle stream, int propertyId, void* data, int dataSize)'''
    return _oniStreamSetProperty(stream, propertyId, data, dataSize)


@oni_call
def oniStreamGetProperty(stream, propertyId, data, pDataSize):
    '''OniStatus oniStreamGetProperty(OniStreamHandle stream, int propertyId, void* data, int* pDataSize)'''
    return _oniStreamGetProperty(stream, propertyId, data, pDataSize)


def oniStreamIsPropertySupported(stream, propertyId):
    '''OniBool oniStreamIsPropertySupported(OniStreamHandle stream, int propertyId)'''
    return _oniStreamIsPropertySupported(stream, propertyId)


@oni_call
def oniStreamInvoke(stream, commandId, data, dataSize):
    '''OniStatus oniStreamInvoke(OniStreamHandle stream, int commandId, void* data, int dataSize)'''
    return _oniStreamInvoke(stream, commandId, data, dataSize)


def oniStreamIsCommandSupported(stream, commandId):
    '''OniBool oniStreamIsCommandSupported(OniStreamHandle stream, int commandId)'''
    return _oniStreamIsCommandSupported(stream, commandId)


@oni_call
def oniStreamSetFrameBuffersAllocator(stream, alloc, free, pCookie):
    '''OniStatus oniStreamSetFrameBuffersAllocator(OniStreamHandle stream, OniFrameAllocBufferCallback alloc, OniFrameFreeBufferCallback free, void* pCookie)'''
    return _oniStreamSetFrameBuffersAllocator(stream, alloc, free, pCookie)


def oniFrameAddRef(pFrame):
    '''void oniFrameAddRef(OniFrame* pFrame)'''
    return _oniFrameAddRef(pFrame)


def oniFrameRelease(pFrame):
    '''void oniFrameRelease(OniFrame* pFrame)'''
    return _oniFrameRelease(pFrame)


@oni_call
def oniCreateRecorder(fileName, pRecorder):
    '''OniStatus oniCreateRecorder(char* fileName, OniRecorderHandle* pRecorder)'''
    return _oniCreateRecorder(fileName, pRecorder)


@oni_call
def oniRecorderAttachStream(recorder, stream, allowLossyCompression):
    '''OniStatus oniRecorderAttachStream(OniRecorderHandle recorder, OniStreamHandle stream, OniBool allowLossyCompression)'''
    return _oniRecorderAttachStream(recorder, stream, allowLossyCompression)


@oni_call
def oniRecorderStart(recorder):
    '''OniStatus oniRecorderStart(OniRecorderHandle recorder)'''
    return _oniRecorderStart(recorder)


def oniRecorderStop(recorder):
    '''void oniRecorderStop(OniRecorderHandle recorder)'''
    return _oniRecorderStop(recorder)


@oni_call
def oniRecorderDestroy(pRecorder):
    '''OniStatus oniRecorderDestroy(OniRecorderHandle* pRecorder)'''
    return _oniRecorderDestroy(pRecorder)


@oni_call
def oniCoordinateConverterDepthToWorld(depthStream, depthX, depthY, depthZ, pWorldX, pWorldY, pWorldZ):
    '''OniStatus oniCoordinateConverterDepthToWorld(OniStreamHandle depthStream, float depthX, float depthY, float depthZ, float* pWorldX, float* pWorldY, float* pWorldZ)'''
    return _oniCoordinateConverterDepthToWorld(depthStream, depthX, depthY, depthZ, pWorldX, pWorldY, pWorldZ)


@oni_call
def oniCoordinateConverterWorldToDepth(depthStream, worldX, worldY, worldZ, pDepthX, pDepthY, pDepthZ):
    '''OniStatus oniCoordinateConverterWorldToDepth(OniStreamHandle depthStream, float worldX, float worldY, float worldZ, float* pDepthX, float* pDepthY, float* pDepthZ)'''
    return _oniCoordinateConverterWorldToDepth(depthStream, worldX, worldY, worldZ, pDepthX, pDepthY, pDepthZ)


@oni_call
def oniCoordinateConverterDepthToColor(depthStream, colorStream, depthX, depthY, depthZ, pColorX, pColorY):
    '''OniStatus oniCoordinateConverterDepthToColor(OniStreamHandle depthStream, OniStreamHandle colorStream, int depthX, int depthY, OniDepthPixel depthZ, int* pColorX, int* pColorY)'''
    return _oniCoordinateConverterDepthToColor(depthStream, colorStream, depthX, depthY, depthZ, pColorX, pColorY)


@oni_call
def oniSetLogOutputFolder(strOutputFolder):
    '''OniStatus oniSetLogOutputFolder(char* strOutputFolder)'''
    return _oniSetLogOutputFolder(strOutputFolder)


@oni_call
def oniGetLogFileName(strFileName, nBufferSize):
    '''OniStatus oniGetLogFileName(char* strFileName, int nBufferSize)'''
    return _oniGetLogFileName(strFileName, nBufferSize)


@oni_call
def oniSetLogMinSeverity(nMinSeverity):
    '''OniStatus oniSetLogMinSeverity(int nMinSeverity)'''
    return _oniSetLogMinSeverity(nMinSeverity)


@oni_call
def oniSetLogConsoleOutput(bConsoleOutput):
    '''OniStatus oniSetLogConsoleOutput(OniBool bConsoleOutput)'''
    return _oniSetLogConsoleOutput(bConsoleOutput)


@oni_call
def oniSetLogFileOutput(bFileOutput):
    '''OniStatus oniSetLogFileOutput(OniBool bFileOutput)'''
    return _oniSetLogFileOutput(bFileOutput)


all_types = [
    OniStatus,
    OniSensorType,
    OniPixelFormat,
    OniDeviceState,
    OniImageRegistrationMode,
    _anon_enum_5,
    OniBool,
    OniCallbackHandleImpl,
    OniCallbackHandle,
    OniVersion,

    AXonLinkSerialNumber,  # AXONLink Struct
    CamExtrinsicParam,
    CamIntrinsicParam,
    AXonLinkCamParam,
    AXonLinkFWVersion,
    AXonLinkSWVersion,
    AxonLinkFirmWarePacketVersion,
    I2cValue,
    E2Reg,
    AXonDSPInterface,
    AXonLinkSetExposureLevel,
    AXonLinkGetExposureLevel,
    AXonCropping,
    AXonLinkReadE2OnType,
    AXonCalibration,
    AXonMotionThreshold,
    AXonBoard_SN,
    AXonSensorExposureWindow,


    AXONLINK,                # AXON_ENUM
    AXonLinkSendFileStatus,

    OniHardwareVersion,
    OniVideoMode,
    OniSensorInfo,
    OniDeviceInfo,
    _OniDevice,
    OniDeviceHandle,
    _OniStream,
    OniStreamHandle,
    _OniRecorder,
    OniRecorderHandle,
    OniFrame,
    OniNewFrameCallback,
    OniGeneralCallback,
    OniDeviceInfoCallback,
    OniDeviceStateCallback,
    OniFrameAllocBufferCallback,
    OniFrameFreeBufferCallback,
    OniDeviceCallbacks,
    OniCropping,
    OniDepthPixel,
    OniGrayscale16Pixel,
    OniGrayscale8Pixel,
    OniRGB888Pixel,
    OniYUV422DoublePixel,
    OniSeek,
    _anon_enum_16,
    _anon_enum_17,
    _anon_enum_18,
    _anon_enum_19,
    XnFWVer,
    XnSensorVer,
    XnHWVer,
    XnChipVer,
    XnCMOSType,
    XnIOImageFormats,
    XnIODepthFormats,
    XnParamResetType,
    XnSensorUsbInterface,
    XnProcessingType,
    XnCroppingMode,
    _anon_enum_28,
    XnFirmwareCroppingMode,
    XnLogFilter,
    XnFilePossibleAttributes,
    XnFlashFileType,
    XnBistType,
    XnBistError,
    XnDepthCMOSType,
    XnImageCMOSType,
    XnSDKVersion,
    XnVersions,
    XnInnerParamData,
    XnDepthAGCBin,
    XnControlProcessingData,
    XnAHBData,
    XnPixelRegistration,
    XnLedState,
    XnCmosBlankingTime,
    XnCmosBlankingUnits,
    XnI2CWriteData,
    XnI2CReadData,
    XnTecData,
    XnTecFastConvergenceData,
    XnEmitterData,
    XnFileAttributes,
    XnParamFileData,
    XnParamFlashData,
    XnFlashFile,
    XnFlashFileList,
    XnProjectorFaultData,
    XnBist,
    _anon_enum_39,
    _anon_enum_40,
    XnUsbInterfaceType,
    XnFwFileVersion,
    XnFwFileFlags,
    XnFwFileEntry,
    XnI2CDeviceInfo,
    XnBistInfo,
    XnFwLogMask,
    XnUsbTestEndpointResult,
    XnCommandAHB,
    XnCommandI2C,
    XnCommandUploadFile,
    XnCommandDownloadFile,
    XnCommandGetFileList,
    XnCommandFormatZone,
    XnCommandDumpEndpoint,
    XnCommandGetI2CDeviceList,
    XnCommandGetBistList,
    XnCommandExecuteBist,
    XnCommandUsbTest,
    XnCommandGetLogMaskList,
    XnCommandSetLogMaskState,
    _anon_enum_41,
    XnFileZone,
    XnBootErrorCode,
    XnFwStreamType,
    XnFwPixelFormat,
    XnFwCompressionType,
    XnDetailedVersion,
    XnBootStatus,
    XnFwStreamInfo,
    XnFwStreamVideoMode,
    XnCommandGetFwStreamList,
    XnCommandCreateStream,
    XnCommandDestroyStream,
    XnCommandStartStream,
    XnCommandStopStream,
    XnCommandGetFwStreamVideoModeList,
    XnCommandSetFwStreamVideoMode,
    XnCommandGetFwStreamVideoMode,
]

all_funcs = [
    oniInitialize,
    oniShutdown,
    oniGetDeviceList,
    oniReleaseDeviceList,
    oniRegisterDeviceCallbacks,
    oniUnregisterDeviceCallbacks,
    oniWaitForAnyStream,
    oniGetVersion,
    oniFormatBytesPerPixel,
    oniGetExtendedError,
    oniDeviceOpen,
    oniDeviceClose,
    oniDeviceGetSensorInfo,
    oniDeviceGetInfo,
    oniDeviceCreateStream,
    oniDeviceEnableDepthColorSync,
    oniDeviceDisableDepthColorSync,
    oniDeviceGetDepthColorSyncEnabled,
    oniDeviceSetProperty,
    oniDeviceGetProperty,
    oniDeviceIsPropertySupported,
    oniDeviceInvoke,
    oniDeviceIsCommandSupported,
    oniDeviceIsImageRegistrationModeSupported,
    oniDeviceOpenEx,
    oniStreamDestroy,
    oniStreamGetSensorInfo,
    oniStreamStart,
    oniStreamStop,
    oniStreamReadFrame,
    oniStreamRegisterNewFrameCallback,
    oniStreamUnregisterNewFrameCallback,
    oniStreamSetProperty,
    oniStreamGetProperty,
    oniStreamIsPropertySupported,
    oniStreamInvoke,
    oniStreamIsCommandSupported,
    oniStreamSetFrameBuffersAllocator,
    oniFrameAddRef,
    oniFrameRelease,
    oniCreateRecorder,
    oniRecorderAttachStream,
    oniRecorderStart,
    oniRecorderStop,
    oniRecorderDestroy,
    oniCoordinateConverterDepthToWorld,
    oniCoordinateConverterWorldToDepth,
    oniCoordinateConverterDepthToColor,
    oniSetLogOutputFolder,
    oniGetLogFileName,
    oniSetLogMinSeverity,
    oniSetLogConsoleOutput,
    oniSetLogFileOutput,
]
