# 设置源文件编码为UTF-8，确保可以处理非ASCII字符
#coding=utf-8
# 导入struct模块，用于处理C语言数据类型的打包和解包
import struct
# 导入sys模块，提供对Python解释器使用或维护的一些变量以及与解释器交互的功能
import sys

#import plc
# 导入os模块，提供了丰富的方法来处理文件和目录
import os
# 将脚本所在目录及其上层目录添加到Python模块搜索路径中，以便于导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('../BerxelSDkDriver/')

# 从BerxelSdkDriver包中导入多个模块，这些模块定义了Berxel Hawk设备的相关接口
from BerxelSdkDriver.BerxelHawkDefines import *
from BerxelSdkDriver.BerxelHawkFrame import *
from BerxelSdkDriver.BerxelHawkDevice import *
from BerxelSdkDriver.BerxelHawkContext import *

# 导入time模块，提供各种操作时间的函数
import time
# 导入datetime模块，提供日期和时间的组合
import datetime
# 导入threading模块，支持多线程编程
import threading
# 导入numpy库，并简化命名为np，用于科学计算
import numpy as np
# 导入opencv-python库，并简化命名为cv2，用于计算机视觉任务
import cv2

# 定义浮点型变量fx, fy, cx, cy，分别用于存储相机内参中的焦距x值、焦距y值、光心x坐标、光心y坐标
fx = 0.0
fy = 0.0
cx = 0.0
cy = 0.0
# 定义布尔型变量bFirst，初值为True，用于某些逻辑判断（在当前代码中未使用）
bFirst = True
# 定义一个名为HawkDepth的类，继承自object，用于管理与Berxel Hawk设备的交互
class HawkDepth(object):
    # 类的构造函数，初始化对象时自动调用
    def __init__(self):
        self.__context = None   # 初始化私有成员变量__context为None
        self.__device = None  # 初始化私有成员变量__device为None
        self.__deviceList = []  # 初始化私有成员变量__deviceList为空列表

    # 打开设备步骤
    # Step 1 open device
    def openDevice(self):

        self.__context = BerxelHawkContext()   # 创建BerxelHawkContext实例

        if self.__context is None:  # 如果创建失败
            print("init failed")  # 打印错误信息
            return  False  # 返回False表示失败

        self.__context.initCamera()  # 初始化相机

        self.__deviceList = self.__context.getDeviceList()  # 获取设备列表

        if len(self.__deviceList) <  1:  # 如果没有找到设备
            print("can not find device")  # 打印错误信息
            return False

        # 尝试打开设备列表中的第一个设备
        self.__device = self.__context.openDevice(self.__deviceList[0])
        # 如果打开设备失败（即self.__device为None）
        if self.__device is None:
            return False
        return True

    # Step 2 ： open Stream
    def startStream(self):
        # 首先检查是否已成功打开设备
        if self.__device is None:
            return  False
        # 设置去噪状态为关闭。这行代码被注释掉了，可能是为了调试或其他用途预留。
        self.__device.setRegistrationEnable(True)
        self.__device.setFrameSync(True)
        self.__device.setSystemClock()
        # self.__device.setDenoiseStatus(False)
        # 声明全局变量fx, fy, cx, cy，这些变量用于存储相机内参
        global  fx, fy, cx ,cy
        # 获取设备内部参数对象
        intrinsicParams  = BerxelHawkDeviceIntrinsicParams()
        # 实际获取设备的内部参数
        intrinsicParams = self.__device.getDeviceIntrinsicParams()

        # 获取当前深度流模式，并设置同样的帧模式
        frameMode = self.__device.getCurrentFrameMode(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'])
        self.__device.setFrameMode(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'] ,frameMode)
        # 启动深度流和彩色流
        ret = self.__device.startStreams(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'] | BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])
        # 根据启动结果返回相应的布尔值
        if ret == 0:
            print("start stream succeed")
            return True
        else:
            print("start stream failed")
            return False

    # 3: read Frame

    def displayImage(self):

        # 从设备读取深度帧，超时时间为30毫秒
        hawkDepthFrame = self.__device.readDepthFrame(50)
        if hawkDepthFrame is None:  # 如果没有成功读取到深度帧
            return 1
        # 从设备读取彩色帧，超时时间为30毫秒
        hawkColorFrame = self.__device.readColorFrame(50)
        if hawkColorFrame is None:  # 如果没有成功读取到彩色帧
            self.__device.releaseFrame(hawkDepthFrame)  # 释放已经读取的深度帧
            return 1

        # 获取深度帧的宽度和高度以及数据缓冲区（作为无符号16位整数）
        depthWidth  = hawkDepthFrame.getWidth()
        depthHeight = hawkDepthFrame.getHeight()
        depthFrameBuffer = hawkDepthFrame.getDataAsUint16()
        # 获取彩色帧的宽度和高度以及数据缓冲区（作为无符号8位整数）
        colorWidth = hawkColorFrame.getWidth()
        colorHeight = hawkColorFrame.getHeight()
        colorFrameBuffer = hawkColorFrame.getDataAsUint8()

        # 将彩色帧的数据转换为NumPy数组，并转换颜色空间为RGB
        color_array = np.ndarray(shape=(colorHeight, colorWidth, 3), dtype=np.uint8, buffer=colorFrameBuffer)
        img = cv2.cvtColor(np.uint8(color_array), cv2.COLOR_BGR2RGB)

        # 将深度帧的数据转换为NumPy数组，并进行缩放以便显示
        depth_array = np.ndarray(shape=(depthHeight, depthWidth), dtype=np.uint16, buffer=depthFrameBuffer)

        # 将深度数据转换为可视化的灰度图
        depth_array_disp = ((depth_array / 10000.) * 255).astype(np.uint8)
        # 使用OpenCV的颜色映射功能将灰度图转换为伪彩色图
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.03), cv2.COLORMAP_JET)
        # 实际获取设备的内部参数
        intrinsicParams = self.__device.getDeviceIntrinsicParams()
        # color_intrinsics 是彩色相机的内参
        fx = intrinsicParams.colorIntrinsicParams.fx/2
        fy = intrinsicParams.colorIntrinsicParams.fy/2
        cx = intrinsicParams.colorIntrinsicParams.cx/2
        cy = intrinsicParams.colorIntrinsicParams.cy/2
        #print(f"fx是{fx}，fy是{fy}，cx是{cx}，cy是{cy}")

        depthFRONT = depth_array >> 3
        depthTail = (depth_array & 0x0007) / 8
        depth_array = depthFRONT + depthTail

        # 创建一个三维坐标数组
        Height, Width = depth_array.shape
        points = np.zeros((Height, Width, 3), dtype=np.float32)
        # 创建坐标网格
        x = np.arange(Width)
        y = np.arange(Height)
        x, y = np.meshgrid(x, y)
        # 计算每个像素点的三维坐标
        Z = depth_array.astype(np.float32)
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        # 将计算结果存储到 points 数组中
        points[:, :, 0] = X
        points[:, :, 1] = Y
        points[:, :, 2] = Z


        # 假设 points 是一个形状为 (height, width, 3) 的三维数组
        points = points.reshape(-1, 3)  # 将三维数组转换为 (N, 3) 形状，其中 N 是点的数量

        # 创建并显示彩色图像窗口
        cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color', img)

        # 创建并显示深度图像窗口
        cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', depth_colormap)
        # 检查是否按下了 's' 键，如果按下了，就保存当前帧的 RGB、深度图和点云
        key = cv2.waitKey(0)
        if key == ord('s'):
            # 保存 RGB 图像
            cv2.imwrite(f'rgb.png', img)
            print('color saved')

            # 保存深度图像
            cv2.imwrite(f'depth.png', depth_colormap)
            print('depth saved')

            #假设有一个三维数据
            array_3d = points
            #保存数组到文件
            #np.save(f'array_3d',array_3d)
            ###################
            ##save .ply##
            """
            此函数将 numpy 数组保存为.ply 文件
            :param vertices: numpy 数组，形状应为 (N, 3) 或 (N, 6)，N 为顶点数量
            :param filename: 要保存的文件名，默认为 'output.ply'
            """
            num_vertices = array_3d.shape[0]

            with open('berxelPoint3D.ply', 'w') as ply_file:
                # 写入文件头
                ply_file.write("ply\n")
                ply_file.write("format ascii 1.0\n")
                ply_file.write(f"element vertex {num_vertices}\n")
                ply_file.write("property float x\n")
                ply_file.write("property float y\n")
                ply_file.write("property float z\n")
                ply_file.write("end_header\n")

                # 写入顶点数据
                for i in range(num_vertices):
                    vertex = array_3d[i]
                    vertex_str = " ".join([str(v) for v in vertex[:3]])
                    ply_file.write(vertex_str + '\n')
                print("cloudpoint saved")
                print("保存成功")
        # 等待按键输入，超时时间为1毫秒
        ch = 0xFF & cv2.waitKey(1)
        # 如果按下的是ESC键、'Q'或'q'键，则返回-1表示退出循环
        if ch == 27 or ch == 81 or ch == 113:
            return -1
        # 释放读取的彩色帧和深度帧
        self.__device.releaseFrame(hawkColorFrame)
        self.__device.releaseFrame(hawkDepthFrame)

        return 1
    def ShowFrame(self):
        print ("Sart Show Frame...")# 打印开始显示帧的信息
        time.sleep(1) # 延迟1秒，可能是为了等待设备初始化完成或其他目的
        while 1: # 进入无限循环以持续显示帧
            ret = self.displayImage() # 调用displayImage方法来显示图像
            if ret != 1:  # 如果displayImage返回值不是1，即发生错误或用户请求退出
                break
        # 关闭流和设备
        self.closeStream()
        self.closeDevice()
        return


    #4 : closeStream
    def closeStream(self):
        if self.__device is None: # 检查设备是否已打开
            return  False
        # 停止深度流和彩色流
        ret = self.__device.stopStream(BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'] | BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'] )
        # 如果返回值为0，表示操作成功
        if ret == 0:
            return True
        else:
            return False

    def closeDevice(self):
        if self.__context is None:  # 检查上下文是否已初始化
            return False
        if self.__device is None:  # 检查设备是否已打开
            return False
        # 关闭设备
        ret = self.__context.closeDevice(self.__device)
        if ret == 0:  # 如果返回值为0，表示操作成功
            print("clsoe device succeed")
        else:
            print("close device Failed")
        # 销毁相机上下文
        self.__context.destroyCamera()

    def StartTest(self):

        if self.openDevice() == False:  # 尝试打开设备
            return
        if self.startStream() == False:  # 尝试启动流
            return
        # 创建一个新线程来显示帧
        tShowFrame = threading.Thread(target=self.ShowFrame)
        tShowFrame.start()  # 启动线程

if __name__ == '__main__':

    depthView = HawkDepth()   # 创建HawkDepth实例
    depthView.StartTest()  # 开始测试
