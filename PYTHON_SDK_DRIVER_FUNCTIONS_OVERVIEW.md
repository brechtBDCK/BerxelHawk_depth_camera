# Berxel Python SDK Driver: Function Overview

This file summarizes the Python API exposed by the driver in `BerxelSdkDriver/`, focused on functions you use to control the camera and read streams.

---

## Quick Usage Flow

1. Create context: `ctx = BerxelHawkContext()`
2. Initialize SDK: `ctx.initCamera()`
3. Discover devices: `devices = ctx.getDeviceList()`
4. Open device: `dev = ctx.openDevice(devices[0])`
5. Configure stream/mode
6. Start streams
7. Read frames
8. Release each frame
9. Stop streams
10. Close device and destroy SDK

So its context -> camera-> device->frame

---

## Stream / Flag Constants You Will Use

- `BERXEL_HAWK_COLOR_STREAM = 0x01`
- `BERXEL_HAWK_DEPTH_STREAM = 0x02`
- `BERXEL_HAWK_IR_STREAM = 0x04`
- Combine streams with bitwise OR, e.g. `0x01 | 0x02` for color+depth.

Stream flag modes:
- `BERXEL_HAWK_SINGULAR_STREAM_FLAG_MODE = 0x01`
- `BERXEL_HAWK_MIX_STREAM_FLAG_MODE = 0x02`
- `BERXEL_HAWK_MIX_HD_STREAM_FLAG_MODE = 0x03`

---

## `BerxelHawkContext` Functions

### `initCamera()`
Initializes the SDK runtime (`berxelInit`).

### `destroyCamera()`
Releases device list resources and destroys the SDK runtime.

### `getDeviceList()`
Scans connected devices and returns depth-camera device entries. Color companions are tracked internally for paired opening.

### `setDeviceStatusCallback(callback, data)`
Registers a device status callback for connect/disconnect events.

### `openDevice(deviceinfo)`
Opens the selected device address and returns a `BerxelHawkDevice` instance.  
If the depth model has a paired color UVC device, the color device is opened too.

### `closeDevice(device)`
Closes the depth device and paired color device handle (if present).

---

## `BerxelHawkDevice` Functions (Camera Control + Streaming)

### `getSupportFrameModes(streamType)`
Returns supported frame modes (`BerxelHawkStreamFrameMode`) for the selected stream.

### `getCurrentFrameMode(streamType)`
Gets the currently active frame mode for the selected stream.

### `setFrameMode(streamType, mode)`
Applies a frame mode (resolution/pixel format/fps) for a stream.

### `setStreamFlagMode(streamFlagMode)`
Sets stream delivery mode (singular/mix/mix-hd).

### `startStreams(streamFlag, callback=None, user=None)`
Opens one or more streams (color/depth/IR).  
If `callback` is provided, stream frames are delivered through native callback mode (`berxelOpenStream2`).

### `stopStream(streamFlag)`
Stops one or more opened streams.

### `readColorFrame(timeout)`
Reads one color frame (blocking up to `timeout` ms). Returns `BerxelHawkFrame` or `None`.

### `readDepthFrame(timeout)`
Reads one depth frame (blocking up to `timeout` ms). Returns `BerxelHawkFrame` or `None`.

### `readIrFrame(timeout)`
Reads one IR frame (blocking up to `timeout` ms). Returns `BerxelHawkFrame` or `None`.

### `releaseFrame(hawkFrame)`
Releases native frame memory for a previously read frame.

### `getVersion()`
Returns SDK/FW/HW version information (`BerxelVersionInfo`) for the current device.

### `getCurrentDeviceInfo()`
Returns current device metadata (`BerxelHawkDeviceInfo`: VID, PID, SN, address, etc.).

### `getDeviceIntrinsicParams()`
Returns camera intrinsics/extrinsics (`BerxelHawkDeviceIntrinsicParams`).

### `setStreamMirror(bMirror)`
Enables/disables stream mirroring (1/0).

### `setRegistrationEnable(bEnable)`
Enables/disables depth-to-color registration.

### `setFrameSync(bEnable)`
Enables/disables frame synchronization.

### `setSystemClock()`
Pushes host system clock to device.

### `setDenoiseStatus(bEnable)`
Enables/disables denoise processing.

### `setTemporalDenoiseStatus(bEnable)`
Enables/disables temporal denoise.

### `setSpatialDenoiseStatus(bEnable)`
Enables/disables spatial denoise.

### `setColorQuality(nValue)`
Sets color stream quality level (driver-defined numeric value).

### `setColorExposureGain(exposureTime, gain)`
Sets manual color exposure and gain values.

### `enableColorAutoExposure()`
Restores/enables color auto exposure.

### `setDepthElectricCurrent(nValue)`
Sets depth illumination electric current (driver-defined numeric value).

### `convertDepthToPoint(pData, width, height, factor, fx, fy, cx, cy, pixelType)`
Converts a depth buffer to a 3D point-cloud structure (`BerxelHawkPoint3DList`).

### `initColorHandle(deviceColorHandle)` (internal helper)
Attaches the paired color-handle when a depth device has a separate color interface.

---

## `BerxelHawkFrame` Functions (Frame Access Helpers)

### `getDataAsUint8()`
Returns frame payload as `uint8` ctypes array (typical for color data).

### `getDataAsUint16()`
Returns frame payload as `uint16` ctypes array (typical for depth/IR data).

### `getOriData()`
Returns the raw pointer to frame payload.

### `getStreamType()`
Returns stream type id for the frame.

### `getFrameIndex()`
Returns incremental frame index.

### `getPixelType()`
Returns pixel format id.

### `getWidth()`
Returns frame width.

### `getHeight()`
Returns frame height.

### `getDataSize()`
Returns payload size in bytes.

### `getTimeStamp()`
Returns frame timestamp.

### `getFps()`
Returns frame-reported fps.

### `getFrameHandle()`
Returns native frame handle pointer.

## Return-Value Notes

- Most control methods return `0` on success and negative/non-zero on failure.
- Frame read methods return `None` on timeout/failure.
- In current SDK code, `setTemporalDenoiseStatus` and `setSpatialDenoiseStatus` do not explicitly return `ret`, so Python returns `None` even though native calls are made.
