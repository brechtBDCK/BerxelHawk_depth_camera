"""Interactive viewer for inspecting saved depth images."""
from pathlib import Path
import sys

import cv2
import numpy as np


WINDOW_NAME = "Depth Inspector"
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 900


def find_latest_depth_image():
    depth_images = sorted(Path("captures_camera").glob("depth_*.png"))
    if not depth_images:
        return None
    return depth_images[-1]


def load_depth_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Failed to load image: {image_path}")
    if image.ndim != 2:
        raise SystemExit(f"Expected a single-channel depth image, got shape {image.shape}")
    return image


def make_display_image(depth_image_mm: np.ndarray):
    nonzero = depth_image_mm[depth_image_mm > 0]
    if nonzero.size == 0:
        scaled = np.zeros_like(depth_image_mm, dtype=np.uint8)
    else:
        # Ignore zeros when scaling so missing pixels do not flatten the preview.
        min_depth = float(nonzero.min())
        max_depth = float(nonzero.max())
        if min_depth == max_depth:
            scaled = np.full(depth_image_mm.shape, 255, dtype=np.uint8)
        else:
            scaled = np.interp(depth_image_mm, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            scaled[depth_image_mm == 0] = 0
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def fit_scale(image_shape):
    height, width = image_shape[:2]
    scale_x = MAX_DISPLAY_WIDTH / width
    scale_y = MAX_DISPLAY_HEIGHT / height
    return min(scale_x, scale_y, 1.0)


def draw_overlay(display_image: np.ndarray, scale: float, x: int, y: int, depth_mm: float):
    overlay = display_image.copy()
    display_x = int(round(x * scale))
    display_y = int(round(y * scale))
    label = f"x={x} y={y} depth_mm={depth_mm:.1f}"

    cv2.circle(overlay, (display_x, display_y), 5, (255, 255, 255), 1)
    cv2.line(overlay, (display_x - 10, display_y), (display_x + 10, display_y), (255, 255, 255), 1)
    cv2.line(overlay, (display_x, display_y - 10), (display_x, display_y + 10), (255, 255, 255), 1)

    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    text_x = min(max(display_x + 12, 8), max(8, overlay.shape[1] - text_width - 8))
    text_y = max(display_y - 12, text_height + 12)

    cv2.rectangle(
        overlay,
        (text_x - 6, text_y - text_height - 8),
        (text_x + text_width + 6, text_y + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        overlay,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_depth_image()
    if image_path is None:
        raise SystemExit("No depth image found. Pass a path like: python3 inspect_depth_image.py captures_camera/depth_0003.png")

    depth_image = load_depth_image(image_path)
    depth_image_mm = depth_image.astype(np.float32)
    scale = fit_scale(depth_image.shape)
    display_image = make_display_image(depth_image_mm)
    if scale != 1.0:
        display_image = cv2.resize(display_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    state = {"x": 0, "y": 0, "needs_redraw": True}

    def on_mouse(event, mouse_x, mouse_y, _flags, _param):
        if event != cv2.EVENT_MOUSEMOVE:
            return
        x = min(max(int(mouse_x / scale), 0), depth_image.shape[1] - 1)
        y = min(max(int(mouse_y / scale), 0), depth_image.shape[0] - 1)
        if (x, y) != (state["x"], state["y"]):
            state["x"] = x
            state["y"] = y
            state["needs_redraw"] = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    nonzero = depth_image_mm[depth_image_mm > 0]
    print(f"Viewing {image_path}")
    print("Move the mouse over the image to inspect depth. Press q or Esc to quit.")
    print("Showing saved depth values directly from the PNG.")
    if nonzero.size:
        print(f"Depth range: min={float(nonzero.min()):.1f} mm max={float(nonzero.max()):.1f} mm")
    else:
        print("Depth range: image contains only zeros.")

    while True:
        if state["needs_redraw"]:
            depth_mm = float(depth_image_mm[state["y"], state["x"]])
            frame = draw_overlay(display_image, scale, state["x"], state["y"], depth_mm)
            cv2.imshow(WINDOW_NAME, frame)
            state["needs_redraw"] = False

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
