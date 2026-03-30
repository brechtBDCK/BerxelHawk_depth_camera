"""
Simple prototype:
- reads depth arrays (.npy) from folder
- uses per-image ROI from config/depth_rois.json
- converts depth to point cloud
- saves one PCD per depth array
- prints timing overview
"""

import glob
import json
import os
import time

import cv2
import numpy as np
import open3d as o3d


STAS_ROOT = "/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS"
DATASET_NAME = "paprika_blue_upright"
DATASET_DIR = os.path.join(STAS_ROOT, DATASET_NAME)

DEPTH_IMAGES_FOLDER = os.path.join(DATASET_DIR, "depth_images")
INTRINSICS_PATH = "/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/config/camera_intrinsics.json"
ROI_JSON_PATH = os.path.join(DATASET_DIR, "depth_rois.json")
LEGACY_ROI_JSON_PATH = "/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/config/depth_rois.json"
OUTPUT_PCD_FOLDER = os.path.join(DATASET_DIR, "pointclouds_generated")

# These depth images are captured with registration enabled in take_images.py.
# That means the saved depth image is depth reprojected into the color camera frame,
# so A2 must use the color intrinsics scaled from the SDK's 1280x800 calibration size
# to the capture size.
SDK_INTRINSICS_SIZE = (1280, 800)
DEPTH_INTRINSICS_KEY = "colorIntrinsicParams"

# Optional voxel downsampling in A2 (per viewpoint cloud). If true then the downsampled clouds will be saved, otherwise the full-resolution clouds will be saved.
ENABLE_VOXEL_DOWNSAMPLING = True
THINNEST_DETAIL_MM = 6.0
VOXEL_SIZE_M = THINNEST_DETAIL_MM / 1000.0 / 4.0


def scale_intrinsics(camera_params, base_size, target_size):
    base_w, base_h = base_size
    target_w, target_h = target_size
    sx = float(target_w) / float(base_w)
    sy = float(target_h) / float(base_h)
    scaled = dict(camera_params)
    scaled["fx"] = float(camera_params["fx"]) * sx
    scaled["fy"] = float(camera_params["fy"]) * sy
    scaled["cx"] = float(camera_params["cx"]) * sx
    scaled["cy"] = float(camera_params["cy"]) * sy
    return scaled


def make_camera_model(camera_params):
    K = np.array(
        [
            [float(camera_params["fx"]), 0.0, float(camera_params["cx"])],
            [0.0, float(camera_params["fy"]), float(camera_params["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array(
        [
            [
                float(camera_params.get("k1", 0.0)),
                float(camera_params.get("k2", 0.0)),
                float(camera_params.get("p1", 0.0)),
                float(camera_params.get("p2", 0.0)),
                float(camera_params.get("k3", 0.0)),
                float(camera_params.get("k4", 0.0)),
                float(camera_params.get("k5", 0.0)),
                float(camera_params.get("k6", 0.0)),
            ]
        ],
        dtype=np.float64,
    )
    return K, dist


def load_scaled_depth_camera_model(config, image_size):
    if DEPTH_INTRINSICS_KEY not in config:
        raise KeyError(
            f"Expected Berxel-style config with '{DEPTH_INTRINSICS_KEY}'. "
            f"Got keys: {sorted(config.keys())}"
        )

    base_w, base_h = SDK_INTRINSICS_SIZE
    image_w, image_h = image_size
    if base_w * image_h != base_h * image_w:
        raise ValueError(
            f"Cannot safely scale depth intrinsics from {base_w}x{base_h} "
            f"to {image_w}x{image_h}: aspect ratio changed."
        )

    scaled_intrinsics = scale_intrinsics(
        config[DEPTH_INTRINSICS_KEY],
        SDK_INTRINSICS_SIZE,
        image_size,
    )
    K, dist = make_camera_model(scaled_intrinsics)
    print(
        "Using registered-depth color intrinsics scaled from "
        f"{base_w}x{base_h} to {image_w}x{image_h}: "
        f"fx={scaled_intrinsics['fx']:.6f}, fy={scaled_intrinsics['fy']:.6f}, "
        f"cx={scaled_intrinsics['cx']:.6f}, cy={scaled_intrinsics['cy']:.6f}"
    )
    return K, dist


def load_depth_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".png":
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    raise ValueError(f"Unsupported depth file extension: {path}")


if __name__ == "__main__":
    t_global = time.perf_counter()

    stage_totals = {
        "load_image": 0.0,
        "build_camera_model": 0.0,
        "undistort_depth": 0.0,
        "backproject": 0.0,
        "voxel_downsample": 0.0,
        "save_pcd": 0.0,
    }

    os.makedirs(OUTPUT_PCD_FOLDER, exist_ok=True)
    print(f"A2 voxel downsampling: {'ON' if ENABLE_VOXEL_DOWNSAMPLING else 'OFF'}")
    if ENABLE_VOXEL_DOWNSAMPLING:
        print(
            f"A2 voxel size: {VOXEL_SIZE_M:.6f} m "
            f"(thinnest_detail_mm={THINNEST_DETAIL_MM}, divisor=4.0)"
        )

    t0 = time.perf_counter()
    with open(INTRINSICS_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    roi_path = ROI_JSON_PATH if os.path.exists(ROI_JSON_PATH) else LEGACY_ROI_JSON_PATH
    with open(roi_path, "r", encoding="utf-8") as f:
        rois = json.load(f)
    depth_paths = sorted(glob.glob(os.path.join(DEPTH_IMAGES_FOLDER, "depth_*.png")))
    if not depth_paths:
        depth_paths = sorted(glob.glob(os.path.join(DEPTH_IMAGES_FOLDER, "*_depth.npy")))
    if not depth_paths:
        raise RuntimeError(f"No supported depth images found in {DEPTH_IMAGES_FOLDER}")
    print(f"[time] load config + rois + list images: {time.perf_counter() - t0:.4f}s")
    print(f"Using ROI file: {roi_path}")
    print(f"Found {len(depth_paths)} depth images")

    saved_count = 0

    for idx, depth_path in enumerate(depth_paths, start=1):
        image_start = time.perf_counter()
        name = os.path.basename(depth_path)
        print(f"\n[{idx}/{len(depth_paths)}] Processing: {name}")

        t0 = time.perf_counter()
        depth_raw = load_depth_image(depth_path)
        if depth_raw is None or depth_raw.size == 0:
            print("Could not read depth array, skipping")
            continue
        if depth_raw.ndim != 2:
            print(f"Expected single-channel depth image, got shape {depth_raw.shape}, skipping")
            continue
        stage = time.perf_counter() - t0
        stage_totals["load_image"] += stage
        print(f"[time] load depth image: {stage:.4f}s")

        h, w = depth_raw.shape

        roi_key = name if name.endswith(".png") else os.path.splitext(name)[0] + ".png"
        roi = rois.get(roi_key, [0, 0, w, h])
        x0, y0, rw, rh = [int(v) for v in roi]
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        rw = max(1, min(rw, w - x0))
        rh = max(1, min(rh, h - y0))
        print(f"ROI: x={x0}, y={y0}, w={rw}, h={rh}")

        t0 = time.perf_counter()
        K, dist = load_scaled_depth_camera_model(config, (w, h))
        stage = time.perf_counter() - t0
        stage_totals["build_camera_model"] += stage
        print(f"[time] build depth camera model: {stage:.4f}s")

        t0 = time.perf_counter()
        map1, map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=K,
            distCoeffs=dist,
            R=np.eye(3, dtype=np.float64),
            newCameraMatrix=K,
            size=(w, h),
            m1type=cv2.CV_32FC1,
        )
        depth_undist = cv2.remap(
            depth_raw,
            map1.astype(np.float32),
            map2.astype(np.float32),
            cv2.INTER_NEAREST,
            None,
            cv2.BORDER_CONSTANT,
            (0.0, 0.0, 0.0, 0.0),
        )
        stage = time.perf_counter() - t0
        stage_totals["undistort_depth"] += stage
        print(f"[time] undistort depth: {stage:.4f}s")

        t0 = time.perf_counter()
        depth_roi = depth_undist[y0 : y0 + rh, x0 : x0 + rw]
        v_rel, u_rel = np.nonzero(depth_roi > 0)
        if u_rel.size == 0:
            print("No valid depth points in ROI, skipping")
            continue

        u = (u_rel + x0).astype(np.float64)
        v = (v_rel + y0).astype(np.float64)
        z = depth_roi[v_rel, u_rel].astype(np.float64) * 0.001
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        points = np.column_stack((x, y, z))
        stage = time.perf_counter() - t0
        stage_totals["backproject"] += stage
        print(f"[time] depth ROI -> 3D backprojection: {stage:.4f}s")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_to_save = pcd
        points_before_voxel = len(pcd_to_save.points)
        saved_mode = "full"

        if ENABLE_VOXEL_DOWNSAMPLING:
            t1 = time.perf_counter()
            pcd_to_save = pcd_to_save.voxel_down_sample(VOXEL_SIZE_M)
            stage = time.perf_counter() - t1
            stage_totals["voxel_downsample"] += stage
            saved_mode = "downsampled"
            print(f"[time] voxel downsample (A2): {stage:.4f}s")
            print(
                f"A2 voxel downsample: {points_before_voxel} -> {len(pcd_to_save.points)} points "
                f"(voxel={VOXEL_SIZE_M:.6f} m)"
            )

        base = os.path.splitext(name)[0].replace("_depth", "")
        output_path = os.path.join(OUTPUT_PCD_FOLDER, f"{base}_points_undist.pcd")
        t0 = time.perf_counter()
        ok = o3d.io.write_point_cloud(
            output_path,
            pcd_to_save,
            write_ascii=False,
            compressed=False,
            print_progress=False,
        )
        if not ok:
            raise RuntimeError(f"Failed to save {output_path}")
        stage = time.perf_counter() - t0
        stage_totals["save_pcd"] += stage
        print(f"[time] save point cloud (.pcd): {stage:.4f}s")
        print(f"Saved: {output_path} ({len(pcd_to_save.points)} points, mode={saved_mode})")

        print(f"[time] image total: {time.perf_counter() - image_start:.4f}s")
        saved_count += 1

    total_time = time.perf_counter() - t_global
    print("\n[timing overview]")
    print(f"[time] images found: {len(depth_paths)}")
    print(f"[time] point clouds saved: {saved_count}")
    print(f"[time] load depth image total: {stage_totals['load_image']:.4f}s")
    print(f"[time] build camera model total: {stage_totals['build_camera_model']:.4f}s")
    print(f"[time] undistort depth total: {stage_totals['undistort_depth']:.4f}s")
    print(f"[time] backprojection total: {stage_totals['backproject']:.4f}s")
    print(f"[time] voxel downsample total: {stage_totals['voxel_downsample']:.4f}s")
    print(f"[time] save pcd total: {stage_totals['save_pcd']:.4f}s")
    print(f"[time] global total: {total_time:.4f}s")
