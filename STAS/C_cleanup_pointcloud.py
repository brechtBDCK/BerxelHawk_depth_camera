"""
Cleanup prototype for one merged point cloud.
Step 1: remove dominant plane/background.
Step 2: keep largest connected voxel component.
No argparse; paths are hardcoded.
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d


STAS_ROOT = Path("/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS")
DATASET_NAME = "paprika_blue_upright"
DATASET_DIR = STAS_ROOT / DATASET_NAME

POINTCLOUD_PATH = DATASET_DIR / "pointclouds_concatenated" / "merged_points.pcd"
CLEANED_POINTCLOUD_PATH = DATASET_DIR / "pointclouds_concatenated" / "merged_points_clean.pcd"

# Plane-removal settings
VOXEL_SIZE_FOR_PLANE_ESTIMATION = 0.003  # meters
RANSAC_DISTANCE_THRESHOLD = 0.003  # meters
RANSAC_N = 3
RANSAC_NUM_ITERATIONS = 500
PLANE_REMOVAL_DISTANCE = 0.005  # meters

# Largest-component settings
COMPONENT_VOXEL_SIZE = 0.006  # meters


def keep_largest_voxel_component(pcd: o3d.geometry.PointCloud, voxel_size: float):
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        return pcd, 0, 0, 0

    min_bound = points.min(axis=0)
    voxel_coords = np.floor((points - min_bound) / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    n_voxels = unique_voxels.shape[0]

    if n_voxels <= 1:
        return pcd, 1, 0, n_voxels

    points_per_voxel = np.bincount(inverse, minlength=n_voxels)
    voxel_index = {tuple(v): i for i, v in enumerate(unique_voxels)}

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    visited = np.zeros(n_voxels, dtype=bool)
    components = []
    component_sizes = []

    for start in range(n_voxels):
        if visited[start]:
            continue

        visited[start] = True
        queue = deque([start])
        component_voxels = []
        point_count = 0

        while queue:
            idx = queue.popleft()
            component_voxels.append(idx)
            point_count += int(points_per_voxel[idx])

            vx, vy, vz = unique_voxels[idx]
            for dx, dy, dz in neighbor_offsets:
                nbr = voxel_index.get((vx + dx, vy + dy, vz + dz))
                if nbr is not None and not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)

        components.append(component_voxels)
        component_sizes.append(point_count)

    largest_id = int(np.argmax(component_sizes))
    keep_voxels = np.zeros(n_voxels, dtype=bool)
    keep_voxels[np.asarray(components[largest_id], dtype=np.int32)] = True
    keep_points = keep_voxels[inverse]
    keep_idx = np.flatnonzero(keep_points).tolist()

    cleaned = pcd.select_by_index(keep_idx)
    removed = points.shape[0] - len(keep_idx)
    return cleaned, len(components), removed, n_voxels


def remove_dominant_plane(pcd: o3d.geometry.PointCloud):
    t0 = time.perf_counter()
    pcd_for_plane = pcd.voxel_down_sample(VOXEL_SIZE_FOR_PLANE_ESTIMATION)
    if pcd_for_plane.is_empty():
        print("Plane step skipped: downsampled cloud is empty.")
        return pcd
    print(f"[time] downsample for plane estimation: {time.perf_counter() - t0:.4f}s")
    print(f"Downsampled points: {len(pcd_for_plane.points)}")

    try:
        t0 = time.perf_counter()
        plane_model, inliers_ds = pcd_for_plane.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_NUM_ITERATIONS,
        )
        print(f"[time] detect dominant plane (RANSAC): {time.perf_counter() - t0:.4f}s")
        if len(inliers_ds) == 0:
            print("Plane step skipped: no dominant plane found.")
            return pcd

        a, b, c, d = plane_model
        normal_norm = float(np.sqrt(a * a + b * b + c * c))
        if normal_norm < 1e-12:
            print("Plane step skipped: invalid plane model.")
            return pcd

        a /= normal_norm
        b /= normal_norm
        c /= normal_norm
        d /= normal_norm
        print(f"Plane model: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

        t0 = time.perf_counter()
        points = np.asarray(pcd.points)
        distances = np.abs(points @ np.array([a, b, c], dtype=np.float64) + d)
        keep_idx = np.flatnonzero(distances > PLANE_REMOVAL_DISTANCE).tolist()
        cleaned = pcd.select_by_index(keep_idx)
        print(f"[time] remove plane points: {time.perf_counter() - t0:.4f}s")
        print(f"Removed points: {len(points) - len(keep_idx)}")
        print(f"Remaining points: {len(cleaned.points)}")
        return cleaned
    except Exception as exc:
        print(f"Plane step skipped: {exc}")
        return pcd


if __name__ == "__main__":
    t_global = time.perf_counter()

    if not POINTCLOUD_PATH.exists():
        raise FileNotFoundError(f"Point cloud not found: {POINTCLOUD_PATH}")

    t0 = time.perf_counter()
    pcd = o3d.io.read_point_cloud(str(POINTCLOUD_PATH))
    if pcd.is_empty():
        raise RuntimeError(f"Input point cloud is empty or unreadable: {POINTCLOUD_PATH}")
    print(f"[time] load point cloud: {time.perf_counter() - t0:.4f}s")
    print(f"Loaded points: {len(pcd.points)}")

    cleaned = remove_dominant_plane(pcd)

    t0 = time.perf_counter()
    cleaned, n_components, removed_spurious, n_voxels = keep_largest_voxel_component(
        cleaned,
        COMPONENT_VOXEL_SIZE,
    )
    print(f"[time] keep largest connected voxel component: {time.perf_counter() - t0:.4f}s")
    print(f"Occupied voxels: {n_voxels}")
    print(f"Connected components found: {n_components}")
    print(f"Removed spurious points: {removed_spurious}")
    print(f"Remaining points: {len(cleaned.points)}")

    CLEANED_POINTCLOUD_PATH.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ok = o3d.io.write_point_cloud(
        str(CLEANED_POINTCLOUD_PATH),
        cleaned,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to save cleaned point cloud: {CLEANED_POINTCLOUD_PATH}")
    print(f"[time] save cleaned point cloud: {time.perf_counter() - t0:.4f}s")
    print(f"Saved: {CLEANED_POINTCLOUD_PATH}")
    print(f"[time] total: {time.perf_counter() - t_global:.4f}s")
