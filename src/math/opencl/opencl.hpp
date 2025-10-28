#ifndef OPENCL_H
#define OPENCL_H

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

  std::string readProgram(const std::string &filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filePath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  cl::Program compileProgram(const std::string &file) {
    std::string source = readProgram(file);
    cl::Program program(context, source);
    try {
      program.build({device});
    } catch (cl::Error &e) {
      std::string build_log =
          program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      std::cerr << "Build log:\n" << build_log << std::endl;
      throw;
    }
    return program;
  }

  void loadPrograms() {
    for (const auto &[programType, filePath] : programPaths) {
      try {
        programs[programType] = compileProgram(filePath);
        std::cout << "Loaded program: " << filePath << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "Failed to load program " << filePath << ": " << e.what()
                  << std::endl;
      }
    }
  }

  void initializeDevice() {
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
    defaultQueue = cl::CommandQueue(context, device);

    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>()
              << "\nPlatform: " << platforms[0].getInfo<CL_PLATFORM_NAME>()
              << "\nCompute units: "
              << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
              << "\nGlobal memory: "
              << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024)
              << " MB" << std::endl;
  }

public:
  OpenCL() {
    try {
      initializeDevice();
      loadPrograms();
    } catch (const cl::Error &e) {
      std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")"
                << std::endl;
      throw;
    }
  }

  OpenCL(const OpenCL &) = delete;
  OpenCL &operator=(const OpenCL &) = delete;
  OpenCL(OpenCL &&) = delete;
  OpenCL &operator=(OpenCL &&) = delete;

  cl::Device &getDevice() { return device; }
  cl::Context &getContext() { return context; }
  cl::CommandQueue &getDefaultQueue() { return defaultQueue; }

  cl::Program &getProgram(Program program) {
    auto it = programs.find(program);
    if (it == programs.end()) {
      throw std::invalid_argument("Program not loaded: " +
                                  std::to_string(static_cast<int>(program)));
    }
    return it->second;
  }

  void printDeviceInfo() const {
    std::cout << "=== OpenCL Device Info ===" << std::endl;
    std::cout << "Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "Version: " << device.getInfo<CL_DEVICE_VERSION>()
              << std::endl;
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

  bool hasProgram(Program program) const {
    return programs.find(program) != programs.end();
  }
};

extern OpenCL openCL;

#endif