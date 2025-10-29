#include "opencl.hpp"

std::string OpenCL::readProgram(const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filePath);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}
cl::Program OpenCL::compileProgram(const std::string &file) {
  std::string source = readProgram(file);
  cl::Program program(context, source);
  try {
    program.build({device});
  } catch (cl::Error &e) {
    std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    std::cerr << "Build log:\n" << build_log << std::endl;
    throw;
  }
  return program;
}
void OpenCL::loadPrograms() {
  for (const auto &entry : programPaths) {
    programs[entry.first] = compileProgram(entry.second);
    std::cout << "Loaded program: " << entry.second << std::endl;
  }
}

void OpenCL::initializeDevice() {
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

OpenCL::OpenCL() {
  try {
    initializeDevice();
    loadPrograms();
  } catch (const cl::Error &e) {
    std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")"
              << std::endl;
    throw;
  }
}

cl::Program &OpenCL::getProgram(Program program) {
  auto it = programs.find(program);
  if (it == programs.end()) {
    throw std::invalid_argument("Program not loaded: " +
                                std::to_string(static_cast<int>(program)));
  }
  return it->second;
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