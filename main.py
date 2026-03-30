"""Minimal one-shot color/depth capture for Berxel camera."""
from pathlib import Path
import re
import time

import cv2
import numpy as np

from BerxelSdkDriver.BerxelHawkContext import BerxelHawkContext
from BerxelSdkDriver.BerxelHawkDefines import BerxelHawkPixelType, BerxelHawkStreamFlagMode, BerxelHawkStreamType

STREAM_WARMUP_SEC = 1.5
READ_TIMEOUT_MS = 1000
DISCARD_INITIAL_FRAMES = 5
DEPTH_PIXEL_TYPE_12I_4D = BerxelHawkPixelType.forward_dict["BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_12I_4D"]


def set_filters(camera):
    camera.setDenoiseStatus(True)
    camera.setTemporalDenoiseStatus(True)
    camera.setSpatialDenoiseStatus(True)
    camera.setRegistrationEnable(True) #takes into account the extrinsics between color and depth sensors
    
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


def next_capture_paths(captures_dir: Path):
    # Keep color/depth indices in lockstep by using a shared max index.
    pattern = re.compile(r"^(?:color|depth)_(\d{4})\.png$")
    max_index = 0
    for path in captures_dir.glob("*.png"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    next_index = max_index + 1
    return (
        captures_dir / f"color_{next_index:04d}.png",
        captures_dir / f"depth_{next_index:04d}.png",
    )


def main(
    warmup_sec: float = STREAM_WARMUP_SEC,
    read_timeout_ms: int = READ_TIMEOUT_MS,
    discard_initial_frames: int = DISCARD_INITIAL_FRAMES,
) -> int:
    total_t0 = time.perf_counter()
    timings = []
    status = 1

    # Create context.
    ctx = BerxelHawkContext()
    camera = None
    color_frame = None
    depth_frame = None

    stream_mode_flag = BerxelHawkStreamFlagMode.forward_dict["BERXEL_HAWK_MIX_STREAM_FLAG_MODE"] 
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

        if camera.setFrameMode(color_stream, modes_color[0]) != 0: #mode is size=1920x1080, fps=5
            timings.append(("setup_camera", time.perf_counter() - t0))
            print("Failed to set color frame mode.")
            return 1
        if camera.setFrameMode(depth_stream, modes_depth[0]) != 0: #mode is size=640x400, fps=5
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
        time.sleep(warmup_sec)
        timings.append(("start_streams_and_warmup", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 3: discard unstable startup frames for both streams.
        for _ in range(discard_initial_frames):
            discard_color = camera.readColorFrame(read_timeout_ms)
            discard_depth = camera.readDepthFrame(read_timeout_ms)
            if discard_color is None or discard_depth is None:
                if discard_color is not None:
                    camera.releaseFrame(discard_color)
                if discard_depth is not None:
                    camera.releaseFrame(discard_depth)
                timings.append(("discard_initial_frames", time.perf_counter() - t0))
                print("Failed while discarding initial color/depth frames.")
                return 1
            camera.releaseFrame(discard_color)
            camera.releaseFrame(discard_depth)
        timings.append(("discard_initial_frames", time.perf_counter() - t0))

        captures_dir = Path("captures_camera")
        captures_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        # Step 4: acquire color frame.
        color_frame = camera.readColorFrame(read_timeout_ms)
        timings.append(("acquire_color_frame", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 5: acquire depth frame.
        depth_frame = camera.readDepthFrame(read_timeout_ms)
        timings.append(("acquire_depth_frame", time.perf_counter() - t0))

        if color_frame is None or depth_frame is None:
            print("Failed to read color/depth frame.")
            return 1
        color_pixel_type = color_frame.getPixelType()
        depth_pixel_type = depth_frame.getPixelType()
        print("color pixel type", color_pixel_type, BerxelHawkPixelType.get(color_pixel_type, "unknown"))
        print("depth pixel type", depth_pixel_type, BerxelHawkPixelType.get(depth_pixel_type, "unknown"))
        print("color size", color_frame.getWidth(), color_frame.getHeight())
        print("depth size", depth_frame.getWidth(), depth_frame.getHeight())

        t0 = time.perf_counter()
        # Step 6: convert color frame buffer to array.
        color_width = color_frame.getWidth()
        color_height = color_frame.getHeight()
        color_array = np.ndarray(shape=(color_height, color_width, 3),dtype=np.uint8,buffer=color_frame.getDataAsUint8()).copy()
        timings.append(("convert_color_buffer", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 7: convert color array BGR->RGB for png output.
        img_color = cv2.cvtColor(np.uint8(color_array), cv2.COLOR_BGR2RGB) #type: ignore
        timings.append(("convert_color_bgr_to_rgb", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 8: convert depth frame buffer to array.
        depth_width = depth_frame.getWidth()
        depth_height = depth_frame.getHeight()
        raw_depth_array = np.ndarray(
            shape=(depth_height, depth_width),
            dtype=np.uint16,
            buffer=depth_frame.getDataAsUint16(),
        ).copy()
        if depth_pixel_type == DEPTH_PIXEL_TYPE_12I_4D:
            print("Depth pixel type is 12I_4D, applying scaling to convert to depth in millimeters.")
            # BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_12I_4D packs 12 integer bits and 4 fractional bits.
            depth_array = (raw_depth_array >> 4).astype(np.float32)
            depth_array += (raw_depth_array & 0x000f).astype(np.float32) / 16.0
            depth_image = np.rint(depth_array).astype(np.uint16)
        else:
            depth_array = raw_depth_array
            depth_image = raw_depth_array
        timings.append(("convert_depth_buffer", time.perf_counter() - t0))

        t0 = time.perf_counter()
        # Step 9: write images to disk.
        color_path, depth_path = next_capture_paths(captures_dir)
        cv2.imwrite(str(color_path), img_color)
        cv2.imwrite(str(depth_path), depth_image)
        timings.append(("save_images", time.perf_counter() - t0))

        status = 0
        return status
    finally:
        t0 = time.perf_counter()
        # Step 10: cleanup resources.
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
