"""Minimal Berxel camera access check."""

from BerxelSdkDriver.BerxelHawkContext import BerxelHawkContext
from BerxelSdkDriver.BerxelHawkDefines import BerxelHawkStreamType, BerxelHawkStreamFlagMode, BerxelHawkDeviceStatus


def _decode_cstr(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    return str(value)


def main():
    ctx = BerxelHawkContext()
    device = None
    depth_flag = BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"]
    
    ctx.initCamera()
    devices = ctx.getDeviceList()
    camera = ctx.openDevice(devices[0])
    if camera is None:
        print("Failed to open camera.")
        return 1


    frame = camera.readDepthFrame(1000)

    print(
        f"Depth frame ok: {frame.getWidth()}x{frame.getHeight()}, "
        f"index={frame.getFrameIndex()}, bytes={frame.getDataSize()}"
    )
    
    camera.releaseFrame(frame)
    camera.stopStream(depth_flag)
    ctx.closeDevice(camera)
    ctx.destroyCamera()


if __name__ == "__main__":
    main()
