#ifndef OPENCL_H
#define OPENCL_H

#include <CL/cl.h>

class OpenCLException : public std::runtime_error {
private:
  cl_int error_code;

public:
  OpenCLException(cl_int error, const std::string &operation)
      : std::runtime_error("Error during " + operation + ": " +
                           std::to_string(error)),
        error_code(error) {}

  cl_int getErrorCode() const { return error_code; }
};

class OpenCL {
public:
  static void checkError(cl_int error, const std::string &operation) {
    if (error != CL_SUCCESS) {
      throw OpenCLException(error, operation);
    }
  }

  static std::string readFile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open kernel file: " + filename);
    }

    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  }
};

#endif