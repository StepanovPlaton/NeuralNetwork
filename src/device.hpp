#ifndef DEVICE_H
#define DEVICE_H

#include <CL/cl.h>

#include "opencl.hpp"

class CalcEngine {
private:
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  std::string device_name;

  void initializeOpenCL() {
    OpenCL::checkError(clGetPlatformIDs(1, &platform, nullptr),
                       "clGetPlatformIDs");
    OpenCL::checkError(
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr),
        "clGetDeviceIDs");

    char name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    device_name = name;

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    if (!context) {
      throw OpenCLException(-1, "clCreateContext");
    }

    std::cout << "OpenCL initialized successfully" << std::endl;
  }

  void cleanup() {
    if (context)
      clReleaseContext(context);
  }

public:
  CalcEngine() { initializeOpenCL(); }

  ~CalcEngine() { cleanup(); }

  const cl_platform_id getPlatform() const { return platform; };
  const cl_device_id getDevice() const { return device; };
  const cl_context getContext() const { return context; };
  const std::string getDeviceName() const { return device_name; };

  void printDeviceInfo() const {
    std::cout << "Using OpenCL device: " << device_name << std::endl;
  }

  cl_mem createBuffer(cl_mem_flags flags, size_t size, void *host_ptr) {
    cl_int ret;
    cl_mem buffer = clCreateBuffer(context, flags, size, host_ptr, &ret);
    OpenCL::checkError(ret, "clCreateBuffer");
    return buffer;
  }

  cl_kernel loadKernel(const std::string &filename) {
    std::string kernelSource = OpenCL::readFile(filename);

    const char *source_str = kernelSource.c_str();
    cl_program program =
        clCreateProgramWithSource(context, 1, &source_str, nullptr, nullptr);
    if (!program) {
      throw OpenCLException(-1, "clCreateProgramWithSource");
    }

    cl_int ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (ret != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), nullptr);

      std::cerr << "Build log:\n" << log.data() << std::endl;
      throw OpenCLException(ret, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mult", nullptr);
    if (!kernel) {
      throw OpenCLException(-1, "clCreateKernel");
    }

    std::cout << "Kernel loaded and compiled successfully" << std::endl;

    return kernel;
  }

  void runKernel(cl_command_queue queue, cl_kernel kernel, int M, int N) {
    size_t globalSize[2] = {static_cast<size_t>(M), static_cast<size_t>(N)};
    OpenCL::checkError(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                              globalSize, nullptr, 0, nullptr,
                                              nullptr),
                       "clEnqueueNDRangeKernel");
  }

  void readResult(cl_command_queue queue, cl_mem buf,
                  std::vector<float> &result) {
    OpenCL::checkError(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0,
                                           result.size() * sizeof(float),
                                           result.data(), 0, nullptr, nullptr),
                       "clEnqueueReadBuffer");
  }

  void setKernelArgs(cl_kernel kernel, cl_mem bufA, cl_mem bufB, cl_mem bufC,
                     int M, int N, int K) {
    OpenCL::checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA),
                       "clSetKernelArg for A");
    OpenCL::checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB),
                       "clSetKernelArg for B");
    OpenCL::checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC),
                       "clSetKernelArg for C");
    OpenCL::checkError(clSetKernelArg(kernel, 3, sizeof(int), &M),
                       "clSetKernelArg for M");
    OpenCL::checkError(clSetKernelArg(kernel, 4, sizeof(int), &N),
                       "clSetKernelArg for N");
    OpenCL::checkError(clSetKernelArg(kernel, 5, sizeof(int), &K),
                       "clSetKernelArg for K");
  }
};

#endif