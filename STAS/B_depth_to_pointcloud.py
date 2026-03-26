"""
Convert undistorted depth/color PNG pairs to colored point clouds.
Input:  captures_camera_undistorted/**/depth_*_undist.png
Input:  captures_camera_undistorted/**/color_*_undist.png
Input:  intrinsics + depth_to_color_extrinsic from intrinsics.json
Output: colored PLY point clouds in calculated_pointclouds/<same-subfolder>/
"""
from pathlib import Path
import json
import time

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "captures_camera_undistorted"
INTRINSICS_PATH = BASE_DIR.parent / "intrinsics.json"
OUTPUT_DIR = BASE_DIR / "calculated_pointclouds"
CALIB_WIDTH = 1280.0
CALIB_HEIGHT = 800.0


def load_calibration():
    with INTRINSICS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ir = data["ir_intrinsic"]
    color = data["color_intrinsic"]
    ext = data["depth_to_color_extrinsic"]
    r_depth_to_color = np.array(ext["rotation_3x3"], dtype=np.float32)
    t_depth_to_color_mm = np.array(
        [ext["translation"]["x"], ext["translation"]["y"], ext["translation"]["z"]],
        dtype=np.float32,
    )
    # Prototype assumption: SDK translation values are in millimeters.
    t_depth_to_color_m = t_depth_to_color_mm * 0.001
    return (
        float(ir["fx"]),
        float(ir["fy"]),
        float(ir["cx"]),
        float(ir["cy"]),
        float(color["fx"]),
        float(color["fy"]),
        float(color["cx"]),
        float(color["cy"]),
        r_depth_to_color,
        t_depth_to_color_m,
    )


def scale_intrinsics_to_image_size(fx, fy, cx, cy, width, height):
    # intrinsics.json values are calibrated at 1280x800; depth images here are 640x400.
    sx = width / CALIB_WIDTH
    sy = height / CALIB_HEIGHT
    return fx * sx, fy * sy, cx * sx, cy * sy


def depth_to_points(depth_mm, fx, fy, cx, cy):
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u].astype(np.float32) * 0.001
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.column_stack((x, y, z))


def colorize_points(points_depth_m, color_bgr, fx, fy, cx, cy, r_depth_to_color, t_depth_to_color):
    points_color_m = (points_depth_m @ r_depth_to_color.T) + t_depth_to_color
    z = points_color_m[:, 2]
    valid_z = z > 0

    u = np.full(len(points_depth_m), -1, dtype=np.int32)
    v = np.full(len(points_depth_m), -1, dtype=np.int32)
    u[valid_z] = np.round(points_color_m[valid_z, 0] * fx / z[valid_z] + cx).astype(np.int32)
    v[valid_z] = np.round(points_color_m[valid_z, 1] * fy / z[valid_z] + cy).astype(np.int32)

    h, w = color_bgr.shape[:2]
    in_frame = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    rgb = np.zeros((len(points_depth_m), 3), dtype=np.uint8)
    if np.any(in_frame):
        bgr = color_bgr[v[in_frame], u[in_frame]]
        rgb[in_frame] = bgr[:, ::-1]

    return rgb, int(np.count_nonzero(in_frame))


def write_ply(path, points, rgb):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        data = np.column_stack((points, rgb))
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d")


def matching_color_path(depth_path):
    return depth_path.with_name(depth_path.name.replace("depth_", "color_", 1))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (
        fx_d,
        fy_d,
        cx_d,
        cy_d,
        fx_c,
        fy_c,
        cx_c,
        cy_c,
        r_depth_to_color,
        t_depth_to_color,
    ) = load_calibration()
    depth_paths = sorted(IMAGES_DIR.rglob("depth_*_undist.png"))

    for depth_path in depth_paths:
        t0 = time.perf_counter()
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise RuntimeError(f"Failed to read depth image: {depth_path}")

        color_path = matching_color_path(depth_path)
        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise RuntimeError(f"Failed to read matching color image: {color_path}")

        h_d, w_d = depth_mm.shape[:2]
        fx_ds, fy_ds, cx_ds, cy_ds = scale_intrinsics_to_image_size(fx_d, fy_d, cx_d, cy_d, w_d, h_d)
        points = depth_to_points(depth_mm, fx_ds, fy_ds, cx_ds, cy_ds)

        h_c, w_c = color_bgr.shape[:2]
        fx_cs, fy_cs, cx_cs, cy_cs = scale_intrinsics_to_image_size(fx_c, fy_c, cx_c, cy_c, w_c, h_c)
        rgb, colored_count = colorize_points(
            points,
            color_bgr,
            fx_cs,
            fy_cs,
            cx_cs,
            cy_cs,
            r_depth_to_color,
            t_depth_to_color,
        )

        rel = depth_path.relative_to(IMAGES_DIR)
        output_dir = OUTPUT_DIR / rel.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{depth_path.stem}_points.ply"
        write_ply(output_path, points, rgb)
        dt = time.perf_counter() - t0
        print(f"{output_path} ({dt:.4f}s, colored {colored_count}/{len(points)} points)")


if __name__ == "__main__":
    main()
