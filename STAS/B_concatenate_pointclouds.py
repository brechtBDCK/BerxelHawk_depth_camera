"""
Concatenate viewpoint point clouds using poses from poses.json.

Important frame convention:
- A2 point clouds are in the color camera frame because depth was captured with registration enabled.
- A1 poses are color-camera-to-board.
- So each cloud only needs the color->board transform.
"""

import glob
import json
import os
import time

import numpy as np
import open3d as o3d


STAS_ROOT = "/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS"
DATASET_NAME = "paprika_blue_upright"
DATASET_DIR = os.path.join(STAS_ROOT, DATASET_NAME)

POINTCLOUD_FOLDER = os.path.join(DATASET_DIR, "pointclouds_generated")
POSES_PATH = "/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/config/poses.json"
INTRINSICS_PATH = "/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/config/camera_intrinsics.json"
OUTPUT_PCD_FOLDER = os.path.join(DATASET_DIR, "pointclouds_concatenated")
OUTPUT_FILENAME = "merged_points.pcd"

# Optional voxel downsampling in B (after merge, before save).
ENABLE_VOXEL_DOWNSAMPLING = True
THINNEST_DETAIL_MM = 6.0
VOXEL_SIZE_M = THINNEST_DETAIL_MM / 1000.0 / 4.0

POSE_KEYS = ("pose_A", "pose_B", "pose_C", "pose_D")
# Keep this explicit so "sorted filenames == pose order" is never ambiguous.
# These are the current STAS captures. The saved depth images are registered, so A2
# outputs color-frame clouds named after the depth images.
CAPTURE_TO_POSE = {
    "depth_0001": "pose_A",
    "depth_0002": "pose_B",
    "depth_0003": "pose_C",
    "depth_0004": "pose_D",
}


def load_poses(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    poses = {}
    for pose_key in POSE_KEYS:
        if pose_key not in raw:
            raise KeyError(f"Missing {pose_key} in {path}")
        T = np.asarray(raw[pose_key], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"{pose_key} must be 4x4, got {T.shape}")
        poses[pose_key] = T
    return poses


def infer_capture_key_from_pcd_path(pcd_path):
    stem = os.path.splitext(os.path.basename(pcd_path))[0]
    if stem.endswith("_points_undist_clean"):
        return stem[: -len("_points_undist_clean")]
    if stem.endswith("_points_undist"):
        return stem[: -len("_points_undist")]
    return stem


if __name__ == "__main__":
    t_global = time.perf_counter()
    stage_totals = {
        "load_poses": 0.0,
        "list_pcd_paths": 0.0,
        "map_captures_to_poses": 0.0,
        "sort_selected": 0.0,
        "read_pcd": 0.0,
        "transform_pcd": 0.0,
        "merge_pcd": 0.0,
        "voxel_downsample": 0.0,
        "mkdir_output": 0.0,
        "save_merged_pcd": 0.0,
    }

    t0 = time.perf_counter()
    poses = load_poses(POSES_PATH)
    stage = time.perf_counter() - t0
    stage_totals["load_poses"] += stage
    print(f"[time] load poses: {stage:.4f}s")

    print(f"B voxel downsampling: {'ON' if ENABLE_VOXEL_DOWNSAMPLING else 'OFF'}")
    if ENABLE_VOXEL_DOWNSAMPLING:
        print(
            f"B voxel size: {VOXEL_SIZE_M:.6f} m "
            f"(thinnest_detail_mm={THINNEST_DETAIL_MM}, divisor=4.0)"
        )

    t0 = time.perf_counter()
    pcd_paths = sorted(glob.glob(f"{POINTCLOUD_FOLDER}/*_points_undist.pcd"))
    stage = time.perf_counter() - t0
    stage_totals["list_pcd_paths"] += stage
    print(f"[time] list input point clouds: {stage:.4f}s")
    if not pcd_paths:
        raise RuntimeError(f"No point clouds found in {POINTCLOUD_FOLDER}")

    t0 = time.perf_counter()
    selected = []
    for pcd_path in pcd_paths:
        capture_key = infer_capture_key_from_pcd_path(pcd_path)
        pose_key = CAPTURE_TO_POSE.get(capture_key)
        if pose_key is None:
            continue
        selected.append((pose_key, pcd_path))
    stage = time.perf_counter() - t0
    stage_totals["map_captures_to_poses"] += stage
    print(f"[time] map captures to poses: {stage:.4f}s")

    if len(selected) != len(POSE_KEYS):
        found = ", ".join(infer_capture_key_from_pcd_path(path) for path in pcd_paths)
        raise RuntimeError(
            f"Expected {len(POSE_KEYS)} mapped captures, found {len(selected)}. "
            f"Available captures: [{found}]"
        )

    t0 = time.perf_counter()
    selected.sort(key=lambda item: POSE_KEYS.index(item[0]))
    stage = time.perf_counter() - t0
    stage_totals["sort_selected"] += stage
    print(f"[time] sort mapped captures: {stage:.4f}s")
    print(f"Using {len(selected)} point clouds with explicit capture->pose mapping.")

    merged = o3d.geometry.PointCloud()

    for idx, (pose_key, pcd_path) in enumerate(selected, start=1):
        cloud_start = time.perf_counter()
        T_color_to_board = poses[pose_key]

        t0 = time.perf_counter()
        T_cloud_to_board = T_color_to_board
        stage = time.perf_counter() - t0
        stage_totals["transform_pcd"] += stage
        print(f"[time] [{idx}/{len(selected)}] build transform: {stage:.4f}s")

        t0 = time.perf_counter()
        pcd = o3d.io.read_point_cloud(pcd_path)
        stage = time.perf_counter() - t0
        stage_totals["read_pcd"] += stage
        print(f"[time] [{idx}/{len(selected)}] read point cloud: {stage:.4f}s")
        if pcd.is_empty():
            raise RuntimeError(f"Point cloud is empty: {pcd_path}")

        t0 = time.perf_counter()
        pcd.transform(T_cloud_to_board)
        stage = time.perf_counter() - t0
        stage_totals["transform_pcd"] += stage
        print(f"[time] [{idx}/{len(selected)}] apply transform: {stage:.4f}s")

        t0 = time.perf_counter()
        merged += pcd
        stage = time.perf_counter() - t0
        stage_totals["merge_pcd"] += stage
        print(f"[time] [{idx}/{len(selected)}] merge cloud: {stage:.4f}s")
        print(
            f"Added {os.path.basename(pcd_path)} as {pose_key} "
            f"with {len(pcd.points)} points"
        )
        print(f"[time] [{idx}/{len(selected)}] cloud total: {time.perf_counter() - cloud_start:.4f}s")

    if ENABLE_VOXEL_DOWNSAMPLING:
        merged_before_voxel = len(merged.points)
        t0 = time.perf_counter()
        merged = merged.voxel_down_sample(VOXEL_SIZE_M)
        voxel_elapsed = time.perf_counter() - t0
        stage_totals["voxel_downsample"] += voxel_elapsed
        print(f"[time] voxel downsample merged cloud (B): {voxel_elapsed:.4f}s")
        print(
            f"B voxel downsample: {merged_before_voxel} -> {len(merged.points)} points "
            f"(voxel={VOXEL_SIZE_M:.6f} m)"
        )

    t0 = time.perf_counter()
    os.makedirs(OUTPUT_PCD_FOLDER, exist_ok=True)
    stage = time.perf_counter() - t0
    stage_totals["mkdir_output"] += stage
    print(f"[time] ensure output directory: {stage:.4f}s")
    output_path = os.path.join(OUTPUT_PCD_FOLDER, OUTPUT_FILENAME)

    t0 = time.perf_counter()
    ok = o3d.io.write_point_cloud(
        output_path,
        merged,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    stage = time.perf_counter() - t0
    stage_totals["save_merged_pcd"] += stage
    print(f"[time] save merged point cloud: {stage:.4f}s")
    if not ok:
        raise RuntimeError(f"Failed to save merged point cloud: {output_path}")

    print(f"Saved merged point cloud: {output_path}")
    print(f"Merged points: {len(merged.points)}")

    total_time = time.perf_counter() - t_global
    print("\n[timing overview]")
    print(f"[time] point clouds found: {len(pcd_paths)}")
    print(f"[time] point clouds merged: {len(selected)}")
    print(f"[time] load poses total: {stage_totals['load_poses']:.4f}s")
    print(f"[time] list input point clouds total: {stage_totals['list_pcd_paths']:.4f}s")
    print(f"[time] map captures to poses total: {stage_totals['map_captures_to_poses']:.4f}s")
    print(f"[time] sort mapped captures total: {stage_totals['sort_selected']:.4f}s")
    print(f"[time] read point clouds total: {stage_totals['read_pcd']:.4f}s")
    print(f"[time] transform point clouds total: {stage_totals['transform_pcd']:.4f}s")
    print(f"[time] merge point clouds total: {stage_totals['merge_pcd']:.4f}s")
    print(f"[time] voxel downsample total: {stage_totals['voxel_downsample']:.4f}s")
    print(f"[time] ensure output directory total: {stage_totals['mkdir_output']:.4f}s")
    print(f"[time] save merged point cloud total: {stage_totals['save_merged_pcd']:.4f}s")
    print(f"[time] global total: {total_time:.4f}s")
