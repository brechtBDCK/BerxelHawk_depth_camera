"""Minimal one-shot color/depth capture for Berxel camera."""
import json
from pathlib import Path
import time

import cv2
import numpy as np

from BerxelSdkDriver.BerxelHawkContext import BerxelHawkContext
from BerxelSdkDriver.BerxelHawkDefines import BerxelHawkStreamFlagMode, BerxelHawkStreamType


def main() -> int:
    # Create context.
    ctx = BerxelHawkContext()
    camera = None
    color_frame = None
    depth_frame = None

    stream_mode_flag = BerxelHawkStreamFlagMode.forward_dict["BERXEL_HAWK_MIX_STREAM_FLAG_MODE"]
    stream_type_flags = (BerxelHawkStreamType.forward_dict["BERXEL_HAWK_COLOR_STREAM"] | BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"])

    try:
        # Initialize context and discover devices.
        ctx.initCamera()
        devices = ctx.getDeviceList()
        if not devices:
            print("No camera found.")
            return 1

        # Open camera.
        camera = ctx.openDevice(devices[0])
        if camera is None:
            print("Failed to open camera.")
            return 1

        # Basic settings for this prototype.
        # Keep defaults simple and explicit.
        # camera.setRegistrationEnable(False)
        # camera.setFrameSync(False)
        # camera.setDenoiseStatus(False)
        # camera.setTemporalDenoiseStatus(False)
        # camera.setSpatialDenoiseStatus(False)

        camera.setStreamFlagMode(stream_mode_flag)

        color_stream = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_COLOR_STREAM"]
        depth_stream = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"]

        color_mode = camera.getCurrentFrameMode(color_stream)
        depth_mode = camera.getCurrentFrameMode(depth_stream)
      
        if camera.setFrameMode(color_stream, color_mode) != 0:
            print("Failed to set color frame mode.")
            return 1
        if camera.setFrameMode(depth_stream, depth_mode) != 0:
            print("Failed to set depth frame mode.")
            return 1

        if camera.startStreams(stream_type_flags) != 0:
            print("Failed to start color/depth streams.")
            return 1

        # Give the streams a moment to produce a frame.
        color_frame = camera.readColorFrame(50)
        depth_frame = camera.readDepthFrame(50)
        if color_frame is None or depth_frame is None:
            print("Failed to read color/depth frame.")
            return 1

        color_width = color_frame.getWidth()
        color_height = color_frame.getHeight()
        color_array = np.ndarray(shape=(color_height, color_width, 3),dtype=np.uint8,buffer=color_frame.getDataAsUint8()).copy()
        #save the array as a png file
        cv2.imwrite("captures/color.png", color_array)
       
        
        depth_width = depth_frame.getWidth()
        depth_height = depth_frame.getHeight()
        depth_array = np.ndarray(shape=(depth_height, depth_width),dtype=np.uint16,buffer=depth_frame.getDataAsUint16()).copy()
        #save the array as a png file
        cv2.imwrite("captures/depth_raw.png", depth_array)
        
        
        depth_array_disp = ((depth_array / 10000.) * 255).astype(np.uint8)
        #save the depth array as a png file
        cv2.imwrite("captures/depth.png", depth_array_disp)

        
            


        return 0
    finally:
        if camera is not None:
            camera.releaseFrame(color_frame)
            camera.releaseFrame(depth_frame)
            camera.stopStream(stream_type_flags)
            ctx.closeDevice(camera)
        ctx.destroyCamera()


if __name__ == "__main__":
    raise SystemExit(main())
