#include "opencl.hpp"

#include <iostream>
#include <stdexcept>

OpenCL::OpenCL() {
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
      throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl::Device> devices;
    bool deviceFound = false;

    for (const auto &platform : platforms) {
      try {
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (!devices.empty()) {
          deviceFound = true;
          break;
        }
      } catch (const cl::Error &) {
        continue;
      }
    }

    if (!deviceFound) {
      for (const auto &platform : platforms) {
        try {
          platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
          if (!devices.empty()) {
            deviceFound = true;
            break;
          }
        } catch (const cl::Error &) {
          continue;
        }
      }
    }

    if (!deviceFound) {
      throw std::runtime_error("No suitable OpenCL devices found");
    }

    device = devices[0];
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device,
                             CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  } catch (const cl::Error &e) {
    std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")"
              << std::endl;
    throw;
  }
}

void OpenCL::printDeviceInfo() const {
  std::cout << "=== OpenCL Device Info ===" << std::endl;
  std::cout << "Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
  std::cout << "Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
  std::cout << "Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
  std::cout << "Compute Units: "
            << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
  std::cout << "Global Memory: "
            << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024)
            << " MB" << std::endl;
  std::cout << "Local Memory: "
            << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB"
            << std::endl;
  std::cout << "Max Work Group Size: "
            << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
}
