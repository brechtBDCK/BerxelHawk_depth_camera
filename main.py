"""Minimal one-shot color/depth capture for Berxel camera."""
from pathlib import Path

import numpy as np

from BerxelSdkDriver.BerxelHawkContext import BerxelHawkContext
from BerxelSdkDriver.BerxelHawkDefines import BerxelHawkPixelType, BerxelHawkStreamType

CAPTURE_DIR = Path("captures")
READ_TIMEOUT_MS = 1500


def write_ppm(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())


def write_pgm_u16(path: Path, img_u16: np.ndarray) -> None:
    h, w = img_u16.shape
    with path.open("wb") as f:
        f.write(f"P5\n{w} {h}\n65535\n".encode("ascii"))
        f.write(img_u16.astype(">u2", copy=False).tobytes())


def depth_to_mm(depth_raw: np.ndarray, pixel_type: int) -> np.ndarray:
    # Pixel type names include integer and decimal bits, e.g. 12I_4D.
    if pixel_type == BerxelHawkPixelType.forward_dict["BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_12I_4D"]:
        return (depth_raw >> 4).astype(np.uint16)
    if pixel_type == BerxelHawkPixelType.forward_dict["BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_13I_3D"]:
        return (depth_raw >> 3).astype(np.uint16)
    return depth_raw.astype(np.uint16, copy=True)


def depth_mm_to_vis(depth_mm: np.ndarray) -> np.ndarray:
    vis = np.zeros_like(depth_mm, dtype=np.uint8)
    max_mm = int(depth_mm.max())
    if max_mm > 0:
        vis = np.clip((depth_mm.astype(np.float32) * 255.0) / float(max_mm), 0, 255).astype(np.uint8)
    return np.stack([vis, vis, vis], axis=2)


def main() -> int:
    ctx = BerxelHawkContext()
    camera = None
    color_frame = None
    depth_frame = None

    color_flag = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_COLOR_STREAM"]
    depth_flag = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"]
    stream_flags = color_flag | depth_flag

    try:
        ctx.initCamera()
        devices = ctx.getDeviceList()
        if not devices:
            print("No camera found.")
            return 1

        camera = ctx.openDevice(devices[0])
        if camera is None:
            print("Failed to open camera.")
            return 1

        if camera.startStreams(stream_flags) != 0:
            print("Failed to start color/depth streams.")
            return 1

        color_frame = camera.readColorFrame(READ_TIMEOUT_MS)
        depth_frame = camera.readDepthFrame(READ_TIMEOUT_MS)
        if color_frame is None or depth_frame is None:
            print("Failed to read one color+depth frame.")
            return 1

        color_w, color_h = color_frame.getWidth(), color_frame.getHeight()
        color_bgr = np.frombuffer(color_frame.getDataAsUint8(), dtype=np.uint8).reshape((color_h, color_w, 3)).copy()
        color_rgb = color_bgr[:, :, ::-1]

        depth_w, depth_h = depth_frame.getWidth(), depth_frame.getHeight()
        depth_raw = np.frombuffer(depth_frame.getDataAsUint16(), dtype=np.uint16).reshape((depth_h, depth_w)).copy()
        depth_mm = depth_to_mm(depth_raw, depth_frame.getPixelType())
        depth_vis_rgb = depth_mm_to_vis(depth_mm)

        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

        (CAPTURE_DIR / "color.raw").write_bytes(color_rgb.tobytes())
        write_ppm(CAPTURE_DIR / "color.ppm", color_rgb)

        (CAPTURE_DIR / "depth.raw").write_bytes(depth_raw.astype("<u2", copy=False).tobytes())
        (CAPTURE_DIR / "depth_mm.raw").write_bytes(depth_mm.astype("<u2", copy=False).tobytes())
        write_pgm_u16(CAPTURE_DIR / "depth.pgm", depth_mm)
        write_pgm_u16(CAPTURE_DIR / "depth_mm.pgm", depth_mm)
        write_ppm(CAPTURE_DIR / "depth.ppm", depth_vis_rgb)
        write_ppm(CAPTURE_DIR / "depth_hist.ppm", depth_vis_rgb)

        print(f"Saved captures in: {CAPTURE_DIR.resolve()}")
        return 0
    finally:
        if camera is not None:
            if color_frame is not None:
                camera.releaseFrame(color_frame)
            if depth_frame is not None:
                camera.releaseFrame(depth_frame)
            camera.stopStream(stream_flags)
            ctx.closeDevice(camera)
        ctx.destroyCamera()


if __name__ == "__main__":
    raise SystemExit(main())
