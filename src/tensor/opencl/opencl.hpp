#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

class OpenCL {
private:
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;

public:
  OpenCL();

  OpenCL(const OpenCL &) = delete;
  OpenCL &operator=(const OpenCL &) = delete;
  OpenCL(OpenCL &&) = delete;
  OpenCL &operator=(OpenCL &&) = delete;

  cl::Device &getDevice() { return device; }
  cl::Context &getContext() { return context; }
  const cl::CommandQueue &getQueue() { return queue; }

  void printDeviceInfo() const;
};

extern OpenCL openCL;
