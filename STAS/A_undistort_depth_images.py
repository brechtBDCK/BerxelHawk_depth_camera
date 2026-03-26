"""
Undistort color and depth images from captures_camera/.

Input:
- captures_camera/color_*.png
- captures_camera/depth_*.png
- intrinsics.json

Output:
- captures_camera_undistorted/color_*_undist.png
- captures_camera_undistorted/depth_*_undist.png
"""
from pathlib import Path
import json

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "captures_camera"
INTRINSICS_PATH = BASE_DIR / "intrinsics.json"
OUTPUT_DIR = BASE_DIR / "captures_camera_undistorted"

# Prototype assumption:
# intrinsics.json appears calibrated at 1280x800, while capture sizes differ.
# We scale fx/fy/cx/cy to each image size directly.
CALIB_WIDTH = 1280.0
CALIB_HEIGHT = 800.0


def scaled_camera_matrix(intrinsic, width, height):
    sx = width / CALIB_WIDTH
    sy = height / CALIB_HEIGHT
    return np.array(
        [
            [float(intrinsic["fx"]) * sx, 0.0, float(intrinsic["cx"]) * sx],
            [0.0, float(intrinsic["fy"]) * sy, float(intrinsic["cy"]) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_vector(intrinsic):
    return np.array(
        [
            float(intrinsic["k1"]),
            float(intrinsic["k2"]),
            float(intrinsic["p1"]),
            float(intrinsic["p2"]),
            float(intrinsic["k3"]),
        ],
        dtype=np.float64,
    )


def undistort_image(image, intrinsic, interpolation):
    h, w = image.shape[:2]
    k = scaled_camera_matrix(intrinsic, w, h)
    d = distortion_vector(intrinsic)
    map1, map2 = cv2.initUndistortRectifyMap(
        k,
        d,
        np.eye(3, dtype=np.float64),
        k,
        (w, h),
        cv2.CV_32FC1,
    )
    return cv2.remap(image, map1, map2, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with INTRINSICS_PATH.open("r", encoding="utf-8") as f:
        intrinsics = json.load(f)

    color_intrinsic = intrinsics["color_intrinsic"]
    depth_intrinsic = intrinsics["ir_intrinsic"]

    for color_path in sorted(INPUT_DIR.glob("color_*.png")):
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        color_undist = undistort_image(color, color_intrinsic, cv2.INTER_LINEAR)
        out_path = OUTPUT_DIR / f"{color_path.stem}_undist.png"
        cv2.imwrite(str(out_path), color_undist)
        print(out_path)

    for depth_path in sorted(INPUT_DIR.glob("depth_*.png")):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth_undist = undistort_image(depth, depth_intrinsic, cv2.INTER_NEAREST)
        out_path = OUTPUT_DIR / f"{depth_path.stem}_undist.png"
        cv2.imwrite(str(out_path), depth_undist)
        print(out_path)


if __name__ == "__main__":
    main()
