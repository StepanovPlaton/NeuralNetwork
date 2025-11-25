#include "opencl.hpp"
#include <CL/opencl.hpp>
#include <format>
#include <string>
#include <unordered_map>

template <typename T, int Dim> class Kernels {
public:
  enum class Vector {
    type2 = 2,
    type4 = 4,
    type8 = 8,
    type16 = 16,
  };
  enum class Method {
    POSITIVE,
    NEGATIVE,
    S_ADD,
    S_MULT,
    T_ADD,
    T_HADAMARD,
    T_MULT,
  };
  constexpr const static std::string type = typeid(T).name();

  // TODO: get native vector size
  static Vector vector = Vector::type8;

private:
  static std::string unaryOperation(std::string name, std::string operation) {
    return std::format(
        R"(
        __kernel void {method}(__global {type}* A, int len) {{
          int gid = get_global_id(0);
          int base = gid * {vector};
          if (base + ({vector}-1) < len) {{
            {type}{vector} data = vload{vector}(gid, A);
            vstore{vector}({operation}data, gid, A);
          }} else {{
            for (int i = 0; i < {vec_size}; i++) {{
              int idx = base + i;
              if (idx < len) A[idx] = {operation}A[idx];
            }}
          }}
        }}
        )",
        std::make_format_args(std::make_pair("method", name),
                              std::make_pair("vector", vector),
                              std::make_pair("type", type),
                              std::make_pair("operation", operation)));
  }

  static std::string scalarOperation(std::string name, std::string operation) {
    return std::format(
        R"(
        __kernel void {method}(__global {type}* A, int len, {type} scalar) {{
          int gid = get_global_id(0);
          int base = gid * {vector};
          if (base + ({vector}-1) < len) {{
            {type}{vector} data = vload{vector}(gid, A);
            data = data {operation} scalar;
            vstore{vector}(data, gid, A);
          }} else {{
            for (int i = 0; i < {vec_size}; i++) {{
              int idx = base + i;
              if (idx < len) A[idx] = A[idx] {operation} scalar;
            }}
          }}
        }}
        )",
        std::make_format_args(std::make_pair("method", name),
                              std::make_pair("vector", vector),
                              std::make_pair("type", type),
                              std::make_pair("operation", operation)));
  }

  static std::string binaryOperation(std::string name, std::string operation) {
    return std::format(
        R"(
        __kernel void {method}(__global {type}* A, __global {type}* B, int len) {{
          int gid = get_global_id(0);
          int base = gid * {vector};
          if (base + ({vector}-1) < len) {{
            {type}{vector} dataA = vload{vector}(gid, A);
            {type}{vector} dataB = vload{vector}(gid, B);
            vstore{vector}(dataA {operation} dataB, gid, A);
          }} else {{
            for (int i = 0; i < {vector}; i++) {{
              int idx = base + i;
              if (idx < len) A[idx] = A[idx] {operation} B[idx];
            }}
          }}
        }}
        )",
        std::make_format_args(std::make_pair("method", name),
                              std::make_pair("vector", vector),
                              std::make_pair("type", type),
                              std::make_pair("operation", operation)));
  }

  static std::unordered_map<Method, std::tuple<std::string, std::string>>
      programs = {
          {Method::POSITIVE, {unaryOperation("positive", "+"), "positive"}},
          {Method::NEGATIVE, {unaryOperation("negative", "-")}, "negative"},

          {Method::S_ADD, {scalarOperation("add", "+")}, "add"},
          {Method::S_MULT, {scalarOperation("mult", "*")}, "mult"},

          {Method::T_ADD, {binaryOperation("add", "+")}, "add"},
          {Method::T_HADAMARD,
           {binaryOperation("hadamard_mult", "*")},
           "hadamard_mult"},
          {Method::T_MULT, {"", "mult"}},
  };

  static inline std::unordered_map<Method, cl::Program> compiledPrograms;
  static inline std::mutex compileMutex;

public:
  static cl::Kernel create(Method method) {
    std::lock_guard<std::mutex> lock(compileMutex);

    auto cache = compiledPrograms.find(method);
    if (cache != compiledPrograms.end()) {
      const auto &programName = std::get<1>(programs[method]);
      return cl::Kernel(cache->second, programName.c_str());
    }

    auto program = programs.find(method);
    if (program == programs.end())
      throw std::runtime_error("Unknown method: " +
                               std::to_string(static_cast<int>(method)));
    const auto &[sourceCode, kernelName] = program->second;

    try {
      cl::Program::Sources sources;
      sources.push_back({sourceCode.c_str(), sourceCode.length()});
      cl::Program program(openCL.getContext(), sources);
      program.build({openCL.getDevice()});
      compiledPrograms[method] = program;
      return cl::Kernel(program, kernelName.c_str());

    } catch (const cl::Error &e) {
      if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
        cl::Program program(openCL.getContext(),
                            {sourceCode.c_str(), sourceCode.length()});
        auto buildInfo =
            program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openCL.getDevice());
        throw std::runtime_error(
            "OpenCL compilation failed: " + std::string(e.what()) +
            "\nBuild log:\n" + buildInfo);
      }
      throw std::runtime_error("OpenCL error: " + std::string(e.what()));
    }
  }
};
