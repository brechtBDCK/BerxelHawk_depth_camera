# coding=utf-8
import os
import sys
import time
from contextlib import redirect_stdout

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append('../BerxelSdkDriver/')

from BerxelSdkDriver.BerxelHawkContext import BerxelHawkContext
from BerxelSdkDriver.BerxelHawkDefines import BerxelHawkStreamFlagMode, BerxelHawkStreamType

FRAME_TIMEOUT_MS = 20
RAW_REFRESH_INTERVAL_SEC = 0.12
RAW_REFRESH_INTERVAL_TEMPORAL_SEC = 0.35
RAW_REFRESH_SCALE_STEP = 1.25
RAW_REFRESH_SCALE_MIN = 0.4
RAW_REFRESH_SCALE_MAX = 4.0
DISPLAY_MIN_MM = 300.0
DISPLAY_MAX_MM = 4000.0
_DEVNULL = open(os.devnull, "w")

# Baseline profile (left panel) keeps filters off.
BASELINE_PROFILE = {
    "stream_mirror": False,
    "registration": False,
    "frame_sync": False,
    "denoise": False,
    "temporal_denoise": False,
    "spatial_denoise": False,
    "color_quality": None,
    "color_exposure_gain": None,
    "color_auto_exposure": False,
}

# Filtered profiles (right panel). Press 1..N or n to switch.
FILTER_PROFILES = [
    (
        "Denoise",
        {
            "stream_mirror": False,
            "registration": False,
            "frame_sync": False,
            "denoise": True,
            "temporal_denoise": False,
            "spatial_denoise": False,
            "color_quality": None,
            "color_exposure_gain": None,
            "color_auto_exposure": False,
        },
    ),
    (
        "Temporal Denoise",
        {
            "stream_mirror": False,
            "registration": False,
            "frame_sync": False,
            "denoise": False,
            "temporal_denoise": True,
            "spatial_denoise": False,
            "color_quality": None,
            "color_exposure_gain": None,
            "color_auto_exposure": False,
        },
    ),
    (
        "Spatial Denoise",
        {
            "stream_mirror": False,
            "registration": False,
            "frame_sync": False,
            "denoise": False,
            "temporal_denoise": False,
            "spatial_denoise": True,
            "color_quality": None,
            "color_exposure_gain": None,
            "color_auto_exposure": False,
        },
    ),
    (
        "Temporal + Spatial",
        {
            "stream_mirror": False,
            "registration": False,
            "frame_sync": False,
            "denoise": False,
            "temporal_denoise": True,
            "spatial_denoise": True,
            "color_quality": None,
            "color_exposure_gain": None,
            "color_auto_exposure": False,
        },
    ),
    (
        "Full Filter Stack",
        {
            "stream_mirror": False,
            "registration": True,
            "frame_sync": True,
            "denoise": True,
            "temporal_denoise": True,
            "spatial_denoise": True,
            "color_quality": None,
            "color_exposure_gain": None,
            "color_auto_exposure": True,
        },
    ),
    (
        "Mirror + Full Stack",
        {
            "stream_mirror": True,
            "registration": True,
            "frame_sync": True,
            "denoise": True,
            "temporal_denoise": True,
            "spatial_denoise": True,
            "color_quality": None,
            "color_exposure_gain": None,
            "color_auto_exposure": True,
        },
    ),
]


def _safe_call(label, fn, *args):
    try:
        # SDK wrapper prints on each set* call; suppress this in the hot path.
        with redirect_stdout(_DEVNULL):
            ret = fn(*args)
    except Exception as exc:
        print(f"[warn] {label} failed: {exc}")
        return

    # SDK uses 0 for success. Some methods currently return None.
    if ret not in (None, 0):
        print(f"[warn] {label} returned {ret}")


def apply_profile(device, profile, push_clock=False, applied_state=None):
    if applied_state is None:
        applied_state = {}

    stream_mirror = bool(profile.get("stream_mirror", False))
    if applied_state.get("stream_mirror") != stream_mirror:
        _safe_call("setStreamMirror", device.setStreamMirror, stream_mirror)
        applied_state["stream_mirror"] = stream_mirror

    registration = bool(profile.get("registration", False))
    if applied_state.get("registration") != registration:
        _safe_call("setRegistrationEnable", device.setRegistrationEnable, registration)
        applied_state["registration"] = registration

    frame_sync = bool(profile.get("frame_sync", False))
    if applied_state.get("frame_sync") != frame_sync:
        _safe_call("setFrameSync", device.setFrameSync, frame_sync)
        applied_state["frame_sync"] = frame_sync

    if push_clock:
        _safe_call("setSystemClock", device.setSystemClock)

    denoise = bool(profile.get("denoise", False))
    if applied_state.get("denoise") != denoise:
        _safe_call("setDenoiseStatus", device.setDenoiseStatus, denoise)
        applied_state["denoise"] = denoise

    temporal_denoise = bool(profile.get("temporal_denoise", False))
    if applied_state.get("temporal_denoise") != temporal_denoise:
        _safe_call("setTemporalDenoiseStatus", device.setTemporalDenoiseStatus, temporal_denoise)
        applied_state["temporal_denoise"] = temporal_denoise

    spatial_denoise = bool(profile.get("spatial_denoise", False))
    if applied_state.get("spatial_denoise") != spatial_denoise:
        _safe_call("setSpatialDenoiseStatus", device.setSpatialDenoiseStatus, spatial_denoise)
        applied_state["spatial_denoise"] = spatial_denoise

    # Optional color controls are available for experiments; keep disabled by default.
    quality = profile.get("color_quality")
    if quality is not None and applied_state.get("color_quality") != quality:
        _safe_call("setColorQuality", device.setColorQuality, int(quality))
        applied_state["color_quality"] = quality

    exposure_gain = profile.get("color_exposure_gain")
    if exposure_gain is not None and applied_state.get("color_exposure_gain") != exposure_gain:
        exposure_time, gain = exposure_gain
        _safe_call("setColorExposureGain", device.setColorExposureGain, int(exposure_time), int(gain))
        applied_state["color_exposure_gain"] = exposure_gain

    color_auto_exposure = bool(profile.get("color_auto_exposure", False))
    if color_auto_exposure and not applied_state.get("color_auto_exposure", False):
        _safe_call("enableColorAutoExposure", device.enableColorAutoExposure)
    applied_state["color_auto_exposure"] = color_auto_exposure


def read_depth_frame_mm(device, timeout_ms=FRAME_TIMEOUT_MS):
    frame = device.readDepthFrame(timeout_ms)
    if frame is None:
        return None

    width = frame.getWidth()
    height = frame.getHeight()
    depth_buffer = frame.getDataAsUint16()
    depth = np.ndarray(shape=(height, width), dtype=np.uint16, buffer=depth_buffer).copy()
    device.releaseFrame(frame)
    return depth


def _to_colormap(depth_mm, lo_mm=DISPLAY_MIN_MM, hi_mm=DISPLAY_MAX_MM):
    scale = max(hi_mm - lo_mm, 1.0)
    clipped = np.clip(depth_mm.astype(np.float32), lo_mm, hi_mm)
    gray = ((clipped - lo_mm) / scale * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def render_compare(raw_depth_mm, filtered_depth_mm, profile_name):
    raw_img = _to_colormap(raw_depth_mm)
    filtered_img = _to_colormap(filtered_depth_mm)

    cv2.putText(raw_img, "RAW", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        filtered_img,
        f"FILTERED: {profile_name}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    panel = np.hstack([raw_img, filtered_img])
    hint = (
        f"q/esc quit | n next | r raw now | - slower raw | = faster raw | "
        f"1-{len(FILTER_PROFILES)} select profile"
    )
    cv2.putText(panel, hint, (16, panel.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
    return panel


def raw_refresh_interval_sec(profile, refresh_scale=1.0):
    # Temporal denoise depends on frame history; frequent raw toggles can destabilize it.
    base_interval = RAW_REFRESH_INTERVAL_SEC
    if bool(profile.get("temporal_denoise", False)):
        base_interval = RAW_REFRESH_INTERVAL_TEMPORAL_SEC
    return base_interval * refresh_scale


def print_profile_help():
    print("Depth compare profiles:")
    for idx, (name, _) in enumerate(FILTER_PROFILES, start=1):
        print(f"  {idx}: {name}")


def run_depth_filter_compare():
    ctx = BerxelHawkContext()
    device = None
    depth_stream = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"]

    try:
        ctx.initCamera()
        devices = ctx.getDeviceList()
        if not devices:
            print("can not find device")
            return 1

        device = ctx.openDevice(devices[0])
        if device is None:
            print("open device failed")
            return 1

        device.setStreamFlagMode(BerxelHawkStreamFlagMode.forward_dict["BERXEL_HAWK_SINGULAR_STREAM_FLAG_MODE"])
        mode = device.getCurrentFrameMode(depth_stream)
        if device.setFrameMode(depth_stream, mode) != 0:
            print("set depth frame mode failed")
            return 1

        if device.startStreams(depth_stream) != 0:
            print("start stream failed")
            return 1

        # One sensor stream cannot output baseline and filtered hardware frames at the same instant.
        # For performance, keep filtered profile active and refresh RAW snapshot periodically.
        applied_state = {}
        apply_profile(device, BASELINE_PROFILE, push_clock=True, applied_state=applied_state)

        profile_index = 0
        print_profile_help()
        print(f"active profile: {FILTER_PROFILES[profile_index][0]}")

        active_name, active_profile = FILTER_PROFILES[profile_index]
        apply_profile(device, active_profile, applied_state=applied_state)
        raw_depth = None
        filtered_depth = None
        force_raw_refresh = True
        next_raw_refresh_ts = 0.0
        raw_refresh_scale = 1.0
        print(
            "RAW refresh controls: '=' faster, '-' slower, 'r' immediate refresh "
            f"(current scale {raw_refresh_scale:.2f}x)"
        )

        while True:
            # Keep filtered profile active for smooth UI.
            filtered_depth = read_depth_frame_mm(device)
            if filtered_depth is None:
                continue

            if raw_depth is None:
                raw_depth = filtered_depth

            panel = render_compare(raw_depth, filtered_depth, active_name)
            cv2.imshow("HawkDepth Filter Compare", panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("n"), ord("N")):
                profile_index = (profile_index + 1) % len(FILTER_PROFILES)
                active_name, active_profile = FILTER_PROFILES[profile_index]
                print(f"active profile: {active_name}")
                apply_profile(device, active_profile, applied_state=applied_state)
                force_raw_refresh = True
                next_raw_refresh_ts = 0.0
                continue
            if ord("1") <= key <= ord(str(len(FILTER_PROFILES))):
                profile_index = key - ord("1")
                active_name, active_profile = FILTER_PROFILES[profile_index]
                print(f"active profile: {active_name}")
                apply_profile(device, active_profile, applied_state=applied_state)
                force_raw_refresh = True
                next_raw_refresh_ts = 0.0
            if key in (ord("r"), ord("R")):
                force_raw_refresh = True
                next_raw_refresh_ts = 0.0
            if key in (ord("-"), ord("_")):
                raw_refresh_scale = min(RAW_REFRESH_SCALE_MAX, raw_refresh_scale * RAW_REFRESH_SCALE_STEP)
                print(f"raw refresh scale: {raw_refresh_scale:.2f}x")
                force_raw_refresh = True
            if key in (ord("="), ord("+")):
                raw_refresh_scale = max(RAW_REFRESH_SCALE_MIN, raw_refresh_scale / RAW_REFRESH_SCALE_STEP)
                print(f"raw refresh scale: {raw_refresh_scale:.2f}x")
                force_raw_refresh = True

            now = time.monotonic()
            if force_raw_refresh or now >= next_raw_refresh_ts:
                apply_profile(device, BASELINE_PROFILE, applied_state=applied_state)
                raw_candidate = read_depth_frame_mm(device)
                apply_profile(device, active_profile, applied_state=applied_state)
                if raw_candidate is not None:
                    raw_depth = raw_candidate
                force_raw_refresh = False
                next_raw_refresh_ts = now + raw_refresh_interval_sec(active_profile, raw_refresh_scale)

        return 0
    finally:
        cv2.destroyAllWindows()
        if device is not None:
            device.stopStream(depth_stream)
            ctx.closeDevice(device)
        ctx.destroyCamera()


if __name__ == "__main__":
    raise SystemExit(run_depth_filter_compare())
