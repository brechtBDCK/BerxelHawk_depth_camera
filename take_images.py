"""One-shot color/depth capture exposed as a single callable function."""
from pathlib import Path
import re
import time

import cv2
import numpy as np

from BerxelSdkDriver.BerxelHawkContext import BerxelHawkContext
from BerxelSdkDriver.BerxelHawkDefines import (
    BerxelHawkPixelType,
    BerxelHawkStreamFlagMode,
    BerxelHawkStreamType,
)


DEPTH_PIXEL_TYPE_12I_4D = BerxelHawkPixelType.forward_dict["BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_12I_4D"]


def next_capture_index(out_dir):
    # Keep color/depth pairs aligned by incrementing one shared numeric index.
    pattern = re.compile(r"^(?:color|depth)_(\d{4})\.png$")
    max_index = 0
    for path in out_dir.glob("*.png"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def take_images(
    captures_dir="captures_camera",
    color_filename=None,
    depth_filename=None,
    warmup_sec=1.5,
    read_timeout_ms=1000,
    discard_initial_frames=5,
    print_timings=True,
):
    """Capture one color frame and one depth frame, then save both to disk."""
    timings = []
    total_t0 = time.perf_counter()

    ctx = BerxelHawkContext()
    camera = None
    color_frame = None
    depth_frame = None

    stream_mode_flag = BerxelHawkStreamFlagMode.forward_dict["BERXEL_HAWK_MIX_STREAM_FLAG_MODE"]
    color_stream = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_COLOR_STREAM"]
    depth_stream = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"]
    stream_type_flags = color_stream | depth_stream

    try:
        t0 = time.perf_counter()
        ctx.initCamera()
        devices = ctx.getDeviceList()
        if not devices:
            raise RuntimeError("No camera found.")

        camera = ctx.openDevice(devices[0])
        if camera is None:
            raise RuntimeError("Failed to open camera.")

        camera.setDenoiseStatus(True)
        camera.setTemporalDenoiseStatus(True)
        camera.setSpatialDenoiseStatus(True)
        camera.setRegistrationEnable(True) #takes into account the extrinsics between color and depth sensors


        camera.setStreamFlagMode(stream_mode_flag)

        modes_color = camera.getSupportFrameModes(color_stream)
        modes_depth = camera.getSupportFrameModes(depth_stream)

        if camera.setFrameMode(color_stream, modes_color[0]) != 0: #mode is size=640x400, fps=5
            raise RuntimeError("Failed to set color frame mode.")
        if camera.setFrameMode(depth_stream, modes_depth[0]) != 0: #mode is size=640x400, fps=5
            raise RuntimeError("Failed to set depth frame mode.")
        timings.append(("setup_camera", time.perf_counter() - t0))

        t0 = time.perf_counter()
        if camera.startStreams(stream_type_flags) != 0:
            raise RuntimeError("Failed to start color/depth streams.")
        time.sleep(warmup_sec)
        timings.append(("start_streams_and_warmup", time.perf_counter() - t0))

        t0 = time.perf_counter()
        for _ in range(discard_initial_frames):
            discard_color = camera.readColorFrame(read_timeout_ms)
            discard_depth = camera.readDepthFrame(read_timeout_ms)
            if discard_color is None or discard_depth is None:
                if discard_color is not None:
                    camera.releaseFrame(discard_color)
                if discard_depth is not None:
                    camera.releaseFrame(discard_depth)
                raise RuntimeError("Failed while discarding initial color/depth frames.")
            camera.releaseFrame(discard_color)
            camera.releaseFrame(discard_depth)
        timings.append(("discard_initial_frames", time.perf_counter() - t0))

        t0 = time.perf_counter()
        color_frame = camera.readColorFrame(read_timeout_ms)
        timings.append(("acquire_color_frame", time.perf_counter() - t0))

        t0 = time.perf_counter()
        depth_frame = camera.readDepthFrame(read_timeout_ms)
        timings.append(("acquire_depth_frame", time.perf_counter() - t0))

        if color_frame is None or depth_frame is None:
            raise RuntimeError("Failed to read color/depth frame.")

        t0 = time.perf_counter()
        color_width = color_frame.getWidth()
        color_height = color_frame.getHeight()
        color_array = np.ndarray(shape=(color_height, color_width, 3),dtype=np.uint8,buffer=color_frame.getDataAsUint8()).copy()

        timings.append(("convert_color_buffer", time.perf_counter() - t0))

        t0 = time.perf_counter()
        img_color = cv2.cvtColor(np.uint8(color_array), cv2.COLOR_BGR2RGB) #type: ignore
        timings.append(("convert_color_bgr_to_rgb", time.perf_counter() - t0))

        t0 = time.perf_counter()
        depth_pixel_type = depth_frame.getPixelType()
        depth_width = depth_frame.getWidth()
        depth_height = depth_frame.getHeight()
        raw_depth_array = np.ndarray(
            shape=(depth_height, depth_width),
            dtype=np.uint16,
            buffer=depth_frame.getDataAsUint16(),
        ).copy()
        if depth_pixel_type == DEPTH_PIXEL_TYPE_12I_4D:
            # BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_12I_4D packs 12 integer bits and 4 fractional bits.
            depth_array = (raw_depth_array >> 4).astype(np.float32)
            depth_array += (raw_depth_array & 0x000f).astype(np.float32) / 16.0
            depth_image = np.rint(depth_array).astype(np.uint16)
        else:
            depth_array = raw_depth_array
            depth_image = raw_depth_array
        timings.append(("convert_depth_buffer", time.perf_counter() - t0))

        t0 = time.perf_counter()
        out_dir = Path(captures_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if color_filename is None or depth_filename is None:
            capture_index = next_capture_index(out_dir)
            color_filename = color_filename or f"color_{capture_index:04d}.png"
            depth_filename = depth_filename or f"depth_{capture_index:04d}.png"
        color_path = out_dir / color_filename
        depth_path = out_dir / depth_filename

        if not cv2.imwrite(str(color_path), img_color):
            raise RuntimeError(f"Failed to write color image: {color_path}")
        if not cv2.imwrite(str(depth_path), depth_image):
            raise RuntimeError(f"Failed to write depth image: {depth_path}")
        timings.append(("save_images", time.perf_counter() - t0))

        return {
            "color_path": str(color_path),
            "depth_path": str(depth_path),
            "color_shape": img_color.shape,
            "depth_shape": depth_array.shape,
            "timings": timings,
        }
    finally:
        cleanup_t0 = time.perf_counter()
        if camera is not None:
            if color_frame is not None:
                camera.releaseFrame(color_frame)
            if depth_frame is not None:
                camera.releaseFrame(depth_frame)
            camera.stopStream(stream_type_flags)
            ctx.closeDevice(camera)
        ctx.destroyCamera()
        timings.append(("cleanup", time.perf_counter() - cleanup_t0))
        timings.append(("total", time.perf_counter() - total_t0))

        if print_timings:
            for name, elapsed in timings:
                print(f"{name}: {elapsed * 1000.0:.3f} ms")
