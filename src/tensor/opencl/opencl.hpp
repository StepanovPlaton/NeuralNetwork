#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <unordered_map>

class OpenCL {
public:
  enum class Method {
    POSITIVE,
    NEGATIVE,
    S_ADD,
    S_MULT,
    T_ADD,
    T_HADAMARD,
    T_MULT,
  };
  enum class Program { ATOMIC, SCALAR, TENSOR, FUSION };

private:
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;

  std::unordered_map<Program, cl::Program> programs;
  std::unordered_map<Program, std::string> programPaths = {
      {Program::ATOMIC, "./opencl/kernels/atomic.cl"},
      {Program::SCALAR, "./opencl/kernels/scalar.cl"},
      {Program::TENSOR, "./opencl/kernels/tensor.cl"},
      {Program::FUSION, "./opencl/kernels/fusion.cl"}};
  std::unordered_map<Method, Program> methodPrograms = {
      {Method::POSITIVE, Program::ATOMIC},
      {Method::NEGATIVE, Program::ATOMIC},
      {Method::S_ADD, Program::SCALAR},
      {Method::S_MULT, Program::SCALAR},
      {Method::T_ADD, Program::TENSOR},
      {Method::T_HADAMARD, Program::TENSOR},
      {Method::T_MULT, Program::TENSOR},
  };
  std::unordered_map<Method, std::string> methodNames = {
      {Method::POSITIVE, "positive"}, {Method::NEGATIVE, "negative"},
      {Method::S_ADD, "add"},         {Method::S_MULT, "mult"},
      {Method::T_ADD, "add"},         {Method::T_HADAMARD, "hadamard_mult"},
      {Method::T_MULT, "mult"},
  };

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
  cl::Kernel createKernel(Method method);

  void printDeviceInfo() const;
};

extern OpenCL openCL;
