"""
input: set of 4 RGB images of a charuco board taken from the same camera, with known camera intrinsics.
output: JSON file with 4 camera-to-board poses (one for each image), 
"""

import json
import os
import ctypes
import ctypes.util

import cv2
import numpy as np

INTRINSICS_PATH = f"/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/config/camera_intrinsics.json"
POSES_OUTPUT_PATH = f"/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/config/poses.json"

# Prototype: fixed image set + fixed output pose keys.
CALIBRATION_IMAGES = [
    f"/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/calibration_images/color_0001.png",
    f"/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/calibration_images/color_0002.png",
    f"/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/calibration_images/color_0003.png",
    f"/home/bdck/PROJECTS_WSL/VAimaging_depth_camera/STAS/calibration_images/color_0004.png",
]
POSE_KEYS = ["pose_A", "pose_B", "pose_C", "pose_D"]

# Prototype: fixed board and camera profile.
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_LENGTH = 0.022
MARKER_LENGTH = 0.016
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MIN_CHARUCO_CORNERS = 6
VISUALIZE_CALIBRATION = True
VIS_WAIT_MS = 0
AXIS_LENGTH_M = 0.08

# Berxel SDK intrinsics in this repo are reported for the camera's 1280x800 profile.
# Our prototype images for A1 are expected to be captured at 640x400, so fx/fy/cx/cy
# must be scaled before pose estimation.
SDK_COLOR_INTRINSICS_SIZE = (1280, 800)
EXPECTED_IMAGE_SIZE = (640, 400)


def save_numpy_dict_to_json(data_dict, filename):
    serializable_dict = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in data_dict.items()
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable_dict, f, indent=4)
    print(f"Saved to {filename}")


def scale_intrinsics(camera_params: dict, base_size, target_size):
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


def make_K_and_dist_from_camera_params(camera_params: dict):
    fx, fy = float(camera_params["fx"]), float(camera_params["fy"])
    cx, cy = float(camera_params["cx"]), float(camera_params["cy"])

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
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


def validate_image_size(image_shape, expected_size):
    width = int(image_shape[1])
    height = int(image_shape[0])
    actual_size = (width, height)
    if actual_size != expected_size:
        raise ValueError(
            f"Expected calibration images at {expected_size[0]}x{expected_size[1]}, "
            f"but got {width}x{height}. Update EXPECTED_IMAGE_SIZE or recapture images."
        )
    return actual_size


def load_scaled_color_camera_model(config: dict, image_size):
    if "colorIntrinsicParams" not in config:
        raise KeyError(
            "Expected Berxel-style config with 'colorIntrinsicParams'. "
            f"Got keys: {sorted(config.keys())}"
        )

    base_intrinsics = config["colorIntrinsicParams"]
    base_w, base_h = SDK_COLOR_INTRINSICS_SIZE
    image_w, image_h = image_size
    if base_w * image_h != base_h * image_w:
        raise ValueError(
            f"Cannot safely scale color intrinsics from {base_w}x{base_h} "
            f"to {image_w}x{image_h}: aspect ratio changed."
        )

    scaled_intrinsics = scale_intrinsics(base_intrinsics, SDK_COLOR_INTRINSICS_SIZE, image_size)
    K, dist = make_K_and_dist_from_camera_params(scaled_intrinsics)
    print(
        "Using color intrinsics scaled from "
        f"{base_w}x{base_h} to {image_w}x{image_h}: "
        f"fx={scaled_intrinsics['fx']:.6f}, fy={scaled_intrinsics['fy']:.6f}, "
        f"cx={scaled_intrinsics['cx']:.6f}, cy={scaled_intrinsics['cy']:.6f}"
    )
    return K, dist


def can_show_gui() -> bool:
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


def visualize_charuco_result(
    image_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_corners,
    marker_ids,
    charuco_corners,
    charuco_ids,
    rvec,
    tvec,
    window_name: str,
):
    vis = image_bgr.copy()
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
    if rvec is not None and tvec is not None:
        cv2.drawFrameAxes(vis, K, dist, rvec, tvec, AXIS_LENGTH_M)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis)
    cv2.waitKey(VIS_WAIT_MS)


def estimate_charuco_extrinsics_from_image(
    image_bgr,
    K,
    dist,
    visualize: bool = False,
    window_name: str = "Calibration visualization",
):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict
    )
    board.setLegacyPattern(True)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    print("Detected markers:", 0 if marker_ids is None else len(marker_ids))
    print("Detected charuco corners:", 0 if charuco_ids is None else len(charuco_ids))

    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
        if visualize:
            visualize_charuco_result(
                image_bgr,
                K,
                dist,
                marker_corners,
                marker_ids,
                charuco_corners,
                charuco_ids,
                None,
                None,
                window_name,
            )
        return None, None, None

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pose_result = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist, rvec, tvec
    )
    ok = pose_result[0] if isinstance(pose_result, tuple) else pose_result

    if not ok:
        if visualize:
            visualize_charuco_result(
                image_bgr,
                K,
                dist,
                marker_corners,
                marker_ids,
                charuco_corners,
                charuco_ids,
                None,
                None,
                window_name,
            )
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)
    T_board_to_cam = np.eye(4)
    T_board_to_cam[:3, :3] = R
    T_board_to_cam[:3, 3] = tvec.flatten()

    if visualize:
        visualize_charuco_result(
            image_bgr,
            K,
            dist,
            marker_corners,
            marker_ids,
            charuco_corners,
            charuco_ids,
            rvec,
            tvec,
            window_name,
        )

    return T_board_to_cam, rvec, tvec


if __name__ == "__main__":
    with open(INTRINSICS_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    first_image = cv2.imread(CALIBRATION_IMAGES[0])
    if first_image is None:
        raise FileNotFoundError(
            f"Calibration image not found or unreadable: {CALIBRATION_IMAGES[0]}"
        )
    image_size = validate_image_size(first_image.shape, EXPECTED_IMAGE_SIZE)
    K, dist = load_scaled_color_camera_model(config, image_size)

    poses = {}
    np.set_printoptions(precision=6, suppress=True)

    use_visualization = VISUALIZE_CALIBRATION and can_show_gui()
    if VISUALIZE_CALIBRATION and not use_visualization:
        print("Visualization requested, but GUI is unavailable in this environment.")

    for image_path, pose_key in zip(CALIBRATION_IMAGES, POSE_KEYS):
        print(f"\nProcessing: {image_path} -> {pose_key}")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Calibration image not found or unreadable: {image_path}")
        validate_image_size(img.shape, EXPECTED_IMAGE_SIZE)

        T, rvec, tvec = estimate_charuco_extrinsics_from_image(
            img,
            K,
            dist,
            visualize=use_visualization,
            window_name=f"Calibration - {pose_key}",
        )
        if T is None:
            print("Pose not found (board not detected or insufficient corners).")
            continue

        print("T_board_to_cam:\n", T)
        R = T[:3, :3]
        t = T[:3, 3:4]
        T_cam_to_board = np.eye(4, dtype=np.float64)
        T_cam_to_board[:3, :3] = R.T
        T_cam_to_board[:3, 3] = (-R.T @ t).reshape(3)
        print("T_cam_to_board:\n", T_cam_to_board)

        poses[pose_key] = T_cam_to_board

    if not poses:
        raise RuntimeError("No poses were estimated.")

    if use_visualization:
        cv2.destroyAllWindows()

    save_numpy_dict_to_json(poses, POSES_OUTPUT_PATH)
