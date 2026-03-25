#!/usr/bin/env python3
import os
import sys

import cv2
import numpy as np

INPUT_DIR = "captures"
OUTPUT_DIR = "captures_converted"
OUTPUT_FORMAT = "png"
DEFAULT_DEPTH_SHIFT_BITS = 2


def usage() -> None:
    print("Usage:")
    print("  python convert_berxel_capture.py")
    print("  python convert_berxel_capture.py [input_dir] [output_dir]")
    print("")
    print("Default:")
    print(f"  input_dir={INPUT_DIR}, output_dir={OUTPUT_DIR}")
    print("")
    print("Converts .ppm/.pgm/.raw files in input_dir into .png files in output_dir.")


def save_image(img: np.ndarray, output_path: str, rgb_input: bool = False) -> None:
    out = img

    if rgb_input and out.ndim == 3 and out.shape[2] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise RuntimeError(f"Failed to write output: {output_path}")


def read_width_height(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read input: {image_path}")
    height, width = img.shape[:2]
    return width, height


def convert_ppm_or_pgm(input_path: str, output_path: str) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read input: {input_path}")
    save_image(img, output_path, rgb_input=False)


def convert_color_raw(input_path: str, output_path: str, width: int, height: int) -> None:
    data = np.fromfile(input_path, dtype=np.uint8)
    expected = width * height * 3
    if data.size != expected:
        raise RuntimeError(f"color.raw size mismatch: expected {expected} bytes, got {data.size}")

    rgb = data.reshape((height, width, 3))
    save_image(rgb, output_path, rgb_input=True)


def convert_depth_raw(input_path: str, output_path: str, width: int, height: int, shift_bits: int) -> None:
    data = np.fromfile(input_path, dtype=np.uint16)
    expected = width * height
    if data.size != expected:
        raise RuntimeError(f"depth.raw size mismatch: expected {expected} uint16 values, got {data.size}")

    depth = data.reshape((height, width))
    if shift_bits > 0:
        depth = (depth >> shift_bits).astype(np.uint16)

    save_image(depth, output_path, rgb_input=False)


def ask(prompt: str, default: str = "") -> str:
    text = f"{prompt}"
    if default:
        text += f" [{default}]"
    text += ": "
    value = input(text).strip()
    return value if value else default


def output_path_for(output_dir: str, input_name: str) -> str:
    stem, ext = os.path.splitext(input_name)
    suffix = ext.lstrip(".")
    out_name = f"{stem}_{suffix}.{OUTPUT_FORMAT}"
    return os.path.join(output_dir, out_name)


def pick_sizes(input_dir: str):
    color_size = None
    depth_size = None

    for name in sorted(os.listdir(input_dir)):
        lower = name.lower()
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue

        if lower.endswith(".ppm") and ("color" in lower) and color_size is None:
            color_size = read_width_height(path)
        elif lower.endswith(".pgm") and ("depth" in lower) and depth_size is None:
            depth_size = read_width_height(path)

    return color_size, depth_size


def convert_all(input_dir: str, output_dir: str) -> int:
    if not os.path.isdir(input_dir):
        print(f"Input folder not found: {input_dir}")
        return 1

    os.makedirs(output_dir, exist_ok=True)

    color_size, depth_size = pick_sizes(input_dir)
    depth_shift_bits = DEFAULT_DEPTH_SHIFT_BITS

    converted = 0
    skipped = 0
    failed = 0

    for name in sorted(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, name)
        if not os.path.isfile(input_path):
            continue

        lower = name.lower()
        ext = os.path.splitext(lower)[1]
        if ext not in (".ppm", ".pgm", ".raw"):
            continue

        output_path = output_path_for(output_dir, name)

        try:
            if ext in (".ppm", ".pgm"):
                convert_ppm_or_pgm(input_path, output_path)
            elif "color" in lower:
                if color_size is None:
                    width = int(ask("Color width", "640"))
                    height = int(ask("Color height", "400"))
                    color_size = (width, height)
                convert_color_raw(input_path, output_path, color_size[0], color_size[1])
            elif "depth" in lower:
                if depth_size is None:
                    width = int(ask("Depth width", "640"))
                    height = int(ask("Depth height", "400"))
                    depth_size = (width, height)
                convert_depth_raw(
                    input_path,
                    output_path,
                    depth_size[0],
                    depth_size[1],
                    depth_shift_bits,
                )
            else:
                print(f"Skip {name}: raw type unclear (expected 'color' or 'depth' in filename)")
                skipped += 1
                continue

            print(f"Saved: {output_path}")
            converted += 1
        except Exception as exc:
            print(f"Error converting {name}: {exc}")
            failed += 1

    print("")
    print(f"Done. converted={converted}, skipped={skipped}, failed={failed}")
    return 1 if failed > 0 else 0



def convert_berxel_capture(input_file_raw: str)-> np.ndarray:
    """takes in the path to a .raw file, and returns the image as both a numpy array and a png"""
    
        
