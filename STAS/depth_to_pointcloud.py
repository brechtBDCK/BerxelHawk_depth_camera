"""
Convert depth PNG images to point clouds.
Input:  undistorted depth captures in captures_camera_undistorted/depth_*_undist.png
Input:  camera intrinsics from intrinsics.json
Output: PLY point clouds in calculated_pointclouds/
"""
from pathlib import Path
import json

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "captures_camera_undistorted"
INTRINSICS_PATH = BASE_DIR / "intrinsics.json"
OUTPUT_DIR = BASE_DIR / "calculated_pointclouds"


def load_intrinsics():
    with INTRINSICS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ir = data["ir_intrinsic"]
    return (
        float(ir["fx"]),
        float(ir["fy"]),
        float(ir["cx"]),
        float(ir["cy"]),
    )


def scale_intrinsics_to_image_size(fx, fy, cx, cy, width, height):
    # intrinsics.json values are calibrated at 1280x800; depth images here are 640x400.
    sx = width / 1280.0
    sy = height / 800.0
    return fx * sx, fy * sy, cx * sx, cy * sy


def depth_to_points(depth_mm, fx, fy, cx, cy):
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u].astype(np.float32) * 0.001
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.column_stack((x, y, z))


def write_ply(path, points):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    fx, fy, cx, cy = load_intrinsics()
    depth_paths = sorted(IMAGES_DIR.glob("depth_*_undist.png"))

    for depth_path in depth_paths:
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        h, w = depth_mm.shape[:2]
        fx_s, fy_s, cx_s, cy_s = scale_intrinsics_to_image_size(fx, fy, cx, cy, w, h)
        points = depth_to_points(depth_mm, fx_s, fy_s, cx_s, cy_s)
        output_path = OUTPUT_DIR / f"{depth_path.stem}_points.ply"
        write_ply(output_path, points)
        print(output_path)


if __name__ == "__main__":
    main()
