"""Minimal one-shot color/depth capture for Berxel camera."""
from pathlib import Path
import time

import cv2
import numpy as np

from BerxelSdkDriver.BerxelHawkContext import BerxelHawkContext
from BerxelSdkDriver.BerxelHawkDefines import BerxelHawkStreamFlagMode, BerxelHawkStreamType

STREAM_WARMUP_SEC = 0.5
READ_TIMEOUT_MS = 200

def set_filters(camera):
    camera.setDenoiseStatus(True)
    camera.setTemporalDenoiseStatus(True)
    camera.setSpatialDenoiseStatus(True)
    
def print_timings_table(timings):
    step_width = max(len("Step"), max((len(name) for name, _ in timings), default=0))
    ms_values = [elapsed * 1000.0 for _, elapsed in timings]
    ms_width = max(len("Time (ms)"), max((len(f"{value:.3f}") for value in ms_values), default=0))
    sep = f"+-{'-' * step_width}-+-{'-' * ms_width}-+"
    print(sep)
    print(f"| {'Step'.ljust(step_width)} | {'Time (ms)'.rjust(ms_width)} |")
    print(sep)
    for name, elapsed in timings:
        print(f"| {name.ljust(step_width)} | {(elapsed * 1000.0):>{ms_width}.3f} |")
    print(sep)

def main() -> int:
    total_t0 = time.perf_counter()
    timings = []
    status = 1

    # Create context.
    ctx = BerxelHawkContext()
    camera = None
    color_frame = None
    depth_frame = None

    stream_mode_flag = BerxelHawkStreamFlagMode.forward_dict["BERXEL_HAWK_MIX_STREAM_FLAG_MODE"] #BERXEL_HAWK_SINGULAR_STREAM_FLAG_MODE or BERXEL_HAWK_MIX_STREAM_FLAG_MODE
    stream_type_flags = (BerxelHawkStreamType.forward_dict["BERXEL_HAWK_COLOR_STREAM"] | BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"])

    try:
        t0 = time.perf_counter()
        # Step 1: discover/open camera and apply streaming config.
        ctx.initCamera()
        devices = ctx.getDeviceList()
        if not devices:
            timings.append(("setup_camera", time.perf_counter() - t0))
            print("No camera found.")
            return 1

        camera = ctx.openDevice(devices[0])
        if camera is None:
            timings.append(("setup_camera", time.perf_counter() - t0))
            print("Failed to open camera.")
            return 1

        set_filters(camera)
        camera.setStreamFlagMode(stream_mode_flag)

        color_stream = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_COLOR_STREAM"]
        depth_stream = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"]
        modes_color = camera.getSupportFrameModes(color_stream)
        modes_depth = camera.getSupportFrameModes(depth_stream)

        if camera.setFrameMode(color_stream, modes_color[17]) != 0:
            timings.append(("setup_camera", time.perf_counter() - t0))
            print("Failed to set color frame mode.")
            return 1
        if camera.setFrameMode(depth_stream, modes_depth[5]) != 0:
            timings.append(("setup_camera", time.perf_counter() - t0))
            print("Failed to set depth frame mode.")
            return 1
        timings.append(("setup_camera", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 2: start streams and wait for warmup.
        if camera.startStreams(stream_type_flags) != 0:
            timings.append(("start_streams_and_warmup", time.perf_counter() - t0))
            print("Failed to start color/depth streams.")
            return 1
        time.sleep(STREAM_WARMUP_SEC)
        timings.append(("start_streams_and_warmup", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 3: acquire color frame.
        color_frame = camera.readColorFrame(READ_TIMEOUT_MS)
        timings.append(("acquire_color_frame", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 4: acquire depth frame.
        depth_frame = camera.readDepthFrame(READ_TIMEOUT_MS)
        timings.append(("acquire_depth_frame", time.perf_counter() - t0))

        if color_frame is None or depth_frame is None:
            print("Failed to read color/depth frame.")
            return 1
        print("depth size", depth_frame.getWidth(), depth_frame.getHeight())

        t0 = time.perf_counter()
        # Step 5: convert color frame buffer to array.
        color_width = color_frame.getWidth()
        color_height = color_frame.getHeight()
        color_array = np.ndarray(shape=(color_height, color_width, 3),dtype=np.uint8,buffer=color_frame.getDataAsUint8()).copy()
        timings.append(("convert_color_buffer", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 6: convert color array BGR->RGB for png output.
        img_color = cv2.cvtColor(np.uint8(color_array), cv2.COLOR_BGR2RGB)
        timings.append(("convert_color_bgr_to_rgb", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 7: convert depth frame buffer to array.
        depth_width = depth_frame.getWidth()
        depth_height = depth_frame.getHeight()
        depth_array = np.ndarray(shape=(depth_height, depth_width),dtype=np.uint16,buffer=depth_frame.getDataAsUint16()).copy()
        timings.append(("convert_depth_buffer", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 8: write images to disk.
        captures_dir = Path("captures")
        captures_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite("captures/color.png", img_color)
        cv2.imwrite("captures/depth.png", depth_array)
        timings.append(("save_images", time.perf_counter() - t0))

        status = 0
        return status
    finally:
        t0 = time.perf_counter()
        # Step 9: cleanup resources.
        if camera is not None:
            camera.releaseFrame(color_frame)
            camera.releaseFrame(depth_frame)
            camera.stopStream(stream_type_flags)
            ctx.closeDevice(camera)
        ctx.destroyCamera()
        timings.append(("cleanup", time.perf_counter() - t0))

        total_elapsed = time.perf_counter() - total_t0
        timings.append(("total", total_elapsed))

        print_timings_table(timings)


if __name__ == "__main__":
    raise SystemExit(main())
