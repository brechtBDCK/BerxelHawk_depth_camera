#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include "BerxelHawkContext.h"
#include "BerxelHawkDevice.h"
#include "BerxelHawkFrame.h"
#include "BerxelHawkDefines.h"
#include "BerxelCommonFunc.h"

namespace {

bool ensureDir(const std::string& path) {
    if (mkdir(path.c_str(), 0755) == 0) {
        return true;
    }
    if (errno == EEXIST) {
        return true;
    }
    std::perror(("mkdir " + path).c_str());
    return false;
}

bool writeBinary(const std::string& path, const uint8_t* data, size_t size) {
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << path << " for writing\n";
        return false;
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    return static_cast<bool>(out);
}

bool writePPM(const std::string& path, const uint8_t* rgb, int width, int height) {
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << path << " for writing\n";
        return false;
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb), static_cast<std::streamsize>(width * height * 3));
    return static_cast<bool>(out);
}

uint16_t depthToMillimeters(uint16_t raw, berxel::BerxelHawkPixelType pixelType) {
    if (pixelType == berxel::BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_12I_4D) {
        return static_cast<uint16_t>(raw >> 4);
    }
    if (pixelType == berxel::BERXEL_HAWK_PIXEL_TYPE_DEP_16BIT_13I_3D) {
        return static_cast<uint16_t>(raw >> 3);
    }
    return static_cast<uint16_t>(raw >> 2);
}

bool writeDepthPGM16(const std::string& path, const uint16_t* depth, int width, int height,
                     berxel::BerxelHawkPixelType pixelType) {
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << path << " for writing\n";
        return false;
    }

    out << "P5\n" << width << " " << height << "\n65535\n";
    for (int i = 0; i < width * height; ++i) {
        uint16_t mm = depthToMillimeters(depth[i], pixelType);
        const uint8_t hi = static_cast<uint8_t>((mm >> 8) & 0xFF);
        const uint8_t lo = static_cast<uint8_t>(mm & 0xFF);
        out.put(static_cast<char>(hi));
        out.put(static_cast<char>(lo));
    }
    return static_cast<bool>(out);
}

bool readDepthFrameOnce(berxel::BerxelHawkDevice* device, berxel::BerxelHawkFrame*& frame) {
    for (int i = 0; i < 60; ++i) {
        frame = nullptr;
        const int ret = device->readDepthFrame(frame, 100);
        if (ret == 0 && frame != nullptr) {
            return true;
        }
    }
    return false;
}

bool readColorFrameOnce(berxel::BerxelHawkDevice* device, berxel::BerxelHawkFrame*& frame) {
    for (int i = 0; i < 60; ++i) {
        frame = nullptr;
        const int ret = device->readColorFrame(frame, 100);
        if (ret == 0 && frame != nullptr) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::string outDir = "captures";
    bool captureDepth = true;
    bool captureColor = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--depth-only") {
            captureColor = false;
        } else if (arg == "--color-only") {
            captureDepth = false;
        } else if (arg == "--out" && i + 1 < argc) {
            outDir = argv[++i];
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--depth-only|--color-only] [--out <dir>]\n";
            return 2;
        }
    }

    if (!captureDepth && !captureColor) {
        std::cerr << "Nothing to capture.\n";
        return 2;
    }

    if (!ensureDir(outDir)) {
        return 1;
    }

    berxel::BerxelHawkContext* context = berxel::BerxelHawkContext::getBerxelContext();
    if (!context) {
        std::cerr << "Failed to create Berxel context\n";
        return 1;
    }

    berxel::BerxelHawkDeviceInfo* devices = nullptr;
    uint32_t deviceCount = 0;
    context->getDeviceList(&devices, &deviceCount);
    if (!devices || deviceCount == 0) {
        std::cerr << "No Berxel device found\n";
        berxel::BerxelHawkContext::destroyBerxelContext(context);
        return 1;
    }

    berxel::BerxelHawkDevice* device = context->openDevice(devices[0]);
    if (!device) {
        std::cerr << "Failed to open Berxel device\n";
        berxel::BerxelHawkContext::destroyBerxelContext(context);
        return 1;
    }

    device->setSystemClock();
    device->setStreamFlagMode(berxel::BERXEL_HAWK_SINGULAR_STREAM_FLAG_MODE);

    bool ok = true;

    if (captureDepth) {
        berxel::BerxelHawkStreamFrameMode depthMode;
        if (device->getCurrentFrameMode(berxel::BERXEL_HAWK_DEPTH_STREAM, &depthMode) == 0) {
            device->setFrameMode(berxel::BERXEL_HAWK_DEPTH_STREAM, &depthMode);
        }

        const int startRet = device->startStreams(berxel::BERXEL_HAWK_DEPTH_STREAM);
        if (startRet != 0) {
            std::cerr << "Failed to start depth stream: " << startRet << "\n";
            ok = false;
        } else {
            berxel::BerxelHawkFrame* depthFrame = nullptr;
            if (!readDepthFrameOnce(device, depthFrame)) {
                std::cerr << "Timed out reading depth frame\n";
                ok = false;
            } else {
                const int w = depthFrame->getWidth();
                const int h = depthFrame->getHeight();
                const auto pixelType = depthFrame->getPixelType();
                const auto* depthData = reinterpret_cast<const uint16_t*>(depthFrame->getData());

                std::string rawPath = outDir + "/depth.raw";
                std::string pgmPath = outDir + "/depth_mm.pgm";
                std::string histPath = outDir + "/depth_hist.ppm";

                writeBinary(rawPath, reinterpret_cast<const uint8_t*>(depthData), depthFrame->getDataSize());
                writeDepthPGM16(pgmPath, depthData, w, h, pixelType);

                std::vector<RGB888> depthRgb(static_cast<size_t>(w) * static_cast<size_t>(h));
                BerxelCommonFunc::getInstance()->convertDepthToRgbByHist(
                    const_cast<uint16_t*>(depthData), depthRgb.data(), w, h, pixelType);
                writePPM(histPath, reinterpret_cast<const uint8_t*>(depthRgb.data()), w, h);

                std::cout << "Saved depth: " << rawPath << ", " << pgmPath << ", " << histPath
                          << " (" << w << "x" << h << ")\n";

                device->releaseFrame(depthFrame);
            }
            device->stopStreams(berxel::BERXEL_HAWK_DEPTH_STREAM);
        }
    }

    if (captureColor) {
        berxel::BerxelHawkStreamFrameMode colorMode;
        if (device->getCurrentFrameMode(berxel::BERXEL_HAWK_COLOR_STREAM, &colorMode) == 0) {
            device->setFrameMode(berxel::BERXEL_HAWK_COLOR_STREAM, &colorMode);
        }

        const int startRet = device->startStreams(berxel::BERXEL_HAWK_COLOR_STREAM);
        if (startRet != 0) {
            std::cerr << "Failed to start color stream: " << startRet << "\n";
            ok = false;
        } else {
            berxel::BerxelHawkFrame* colorFrame = nullptr;
            if (!readColorFrameOnce(device, colorFrame)) {
                std::cerr << "Timed out reading color frame\n";
                ok = false;
            } else {
                const int w = colorFrame->getWidth();
                const int h = colorFrame->getHeight();
                const uint8_t* colorData = reinterpret_cast<const uint8_t*>(colorFrame->getData());

                std::string rawPath = outDir + "/color.raw";
                std::string ppmPath = outDir + "/color.ppm";

                writeBinary(rawPath, colorData, colorFrame->getDataSize());
                writePPM(ppmPath, colorData, w, h);

                std::cout << "Saved color: " << rawPath << ", " << ppmPath
                          << " (" << w << "x" << h << ")\n";

                device->releaseFrame(colorFrame);
            }
            device->stopStreams(berxel::BERXEL_HAWK_COLOR_STREAM);
        }
    }

    context->closeDevice(device);
    berxel::BerxelHawkContext::destroyBerxelContext(context);

    return ok ? 0 : 1;
}
