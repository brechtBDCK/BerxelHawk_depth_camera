"""
Throwaway helper script:
- asks ROI for each of 4 depth images
- saves ROIs to JSON for A2_rgbd_to_pointcloud.py
"""

import glob
import json
import os
import ctypes
import ctypes.util

import cv2
import numpy as np


STAS_ROOT = "/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS"
DATASET_NAME = "paprika_grey_upright"
DATASET_DIR = os.path.join(STAS_ROOT, DATASET_NAME)

DEPTH_IMAGES_FOLDER = os.path.join(DATASET_DIR, "depth_images")
ROI_JSON_PATH = os.path.join(DATASET_DIR, "depth_rois.json")
NUM_IMAGES = 4


def can_show_gui():
    if os.name == "nt":
        return True

    x11_display = os.environ.get("DISPLAY")
    if not x11_display:
        return False

    lib_name = ctypes.util.find_library("X11")
    if not lib_name:
        return False

    try:
        lib_x11 = ctypes.cdll.LoadLibrary(lib_name)
        lib_x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        lib_x11.XOpenDisplay.restype = ctypes.c_void_p
        lib_x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display_handle = lib_x11.XOpenDisplay(x11_display.encode("utf-8"))
        if not display_handle:
            return False
        lib_x11.XCloseDisplay(display_handle)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    depth_paths = sorted(glob.glob(os.path.join(DEPTH_IMAGES_FOLDER, "*.png")))[:NUM_IMAGES]
    if len(depth_paths) < NUM_IMAGES:
        raise RuntimeError(
            f"Expected {NUM_IMAGES} depth images in {DEPTH_IMAGES_FOLDER}, found {len(depth_paths)}"
        )

    rois = {}

    for depth_path in depth_paths:
        name = os.path.basename(depth_path)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"Could not read {depth_path}")

        h, w = depth_raw.shape
        print(f"\nSelect ROI for {name}")

        valid = depth_raw > 0
        if np.any(valid):
            lo = np.percentile(depth_raw[valid], 2)
            hi = np.percentile(depth_raw[valid], 98)
            if hi <= lo:
                hi = lo + 1.0
            vis_gray = np.clip(depth_raw.astype(np.float32), lo, hi)
            vis_gray = ((vis_gray - lo) * (255.0 / (hi - lo))).astype(np.uint8)
        else:
            vis_gray = np.zeros_like(depth_raw, dtype=np.uint8)
        vis = cv2.applyColorMap(vis_gray, cv2.COLORMAP_TURBO)

        roi = None
        if can_show_gui():
            try:
                window_name = f"ROI: {name} (ENTER confirm, C cancel)"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                x, y, rw, rh = map(
                    int,
                    cv2.selectROI(window_name, vis, showCrosshair=True, fromCenter=False),
                )
                cv2.destroyWindow(window_name)
                if rw > 0 and rh > 0:
                    roi = [x, y, rw, rh]
            except Exception:
                roi = None

        if roi is None:
            user = input(
                "Enter ROI as x,y,w,h (or press Enter for full image): "
            ).strip()
            if user:
                x_str, y_str, rw_str, rh_str = [p.strip() for p in user.split(",")]
                roi = [int(x_str), int(y_str), int(rw_str), int(rh_str)]
            else:
                roi = [0, 0, w, h]

        x, y, rw, rh = roi
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        roi = [x, y, rw, rh]

        rois[name] = roi
        print(f"Saved ROI for {name}: {roi}")

    with open(ROI_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2)

    print(f"\nSaved ROI JSON: {ROI_JSON_PATH}")
