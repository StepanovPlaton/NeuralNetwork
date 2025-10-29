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
  enum class Program { MATRIX, MATH, IMAGE_PROCESSING };

private:
  cl::Device device;
  cl::Context context;
  cl::CommandQueue defaultQueue;

  std::unordered_map<Program, cl::Program> programs;
  std::unordered_map<Program, std::string> programPaths = {
      {Program::MATRIX, "./kernels/matrix.cl"}};

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
  cl::CommandQueue &getDefaultQueue() { return defaultQueue; }

  cl::Program &getProgram(Program program);
  void printDeviceInfo() const;
};

extern OpenCL openCL;
