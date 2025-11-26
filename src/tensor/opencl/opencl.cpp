#include "opencl.hpp"

#include <iostream>
#include <stdexcept>

OpenCL::OpenCL() {}

void OpenCL::init() {
  try {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
      throw std::runtime_error("No OpenCL platforms found");
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
    if (!deviceFound)
      throw std::runtime_error("No suitable OpenCL devices found");
    device = devices[0];
    printDeviceInfo();
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
  std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();

  std::cout << "Optimal vector sizes:" << std::endl;
  try {
    cl_uint short_native =
        device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT>();
    cl_uint short_preferred =
        device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>();
    std::cout << "  short: native=" << short_native
              << ", preferred=" << short_preferred << std::endl;
  } catch (const cl::Error &e) {
    std::cout << "  short: N/A (error: " << e.what() << ")" << std::endl;
  }
  try {
    cl_uint int_native = device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>();
    cl_uint int_preferred =
        device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>();
    std::cout << "  int: native=" << int_native
              << ", preferred=" << int_preferred << std::endl;
  } catch (const cl::Error &e) {
    std::cout << "  int: N/A (error: " << e.what() << ")" << std::endl;
  }
  try {
    if (extensions.find("cl_khr_fp16") != std::string::npos) {
      cl_uint half_native =
          device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF>();
      cl_uint half_preferred =
          device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>();
      std::cout << "  half: native=" << half_native
                << ", preferred=" << half_preferred << std::endl;
    } else {
      std::cout << "  half: not supported" << std::endl;
    }
  } catch (const cl::Error &e) {
    std::cout << "  half: N/A (error: " << e.what() << ")" << std::endl;
  }
  try {
    cl_uint float_native =
        device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>();
    cl_uint float_preferred =
        device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
    std::cout << "  float: native=" << float_native
              << ", preferred=" << float_preferred << std::endl;
  } catch (const cl::Error &e) {
    std::cout << "  float: N/A (error: " << e.what() << ")" << std::endl;
  }
  try {
    if (extensions.find("cl_khr_fp64") != std::string::npos ||
        device.getInfo<CL_DEVICE_VERSION>().find("1.0") == std::string::npos) {
      cl_uint double_native =
          device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
      cl_uint double_preferred =
          device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>();
      std::cout << "  double: native=" << double_native
                << ", preferred=" << double_preferred << std::endl;
    } else {
      std::cout << "  double: not supported" << std::endl;
    }
  } catch (const cl::Error &e) {
    std::cout << "  double: N/A (error: " << e.what() << ")" << std::endl;
  }
}
