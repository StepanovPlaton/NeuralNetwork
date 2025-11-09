#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

class OpenCL {
public:
  enum class Program { TENSOR };

private:
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;

  std::unordered_map<Program, cl::Program> programs;
  std::unordered_map<Program, std::string> programPaths = {
      {Program::TENSOR, "./opencl/kernels/tensor.cl"}};

  std::string readProgram(const std::string &filePath);
  cl::Program compileProgram(const std::string &file);
  void loadPrograms();

  void initializeDevice();

public:
  OpenCL();

  OpenCL(const OpenCL &) = delete;
  OpenCL &operator=(const OpenCL &) = delete;
  OpenCL(OpenCL &&) = delete;
  OpenCL &operator=(OpenCL &&) = delete;

  cl::Device &getDevice() { return device; }
  cl::Context &getContext() { return context; }
  const cl::CommandQueue &getQueue() { return queue; }

  cl::Program &getProgram(Program program);
  void printDeviceInfo() const;
};
