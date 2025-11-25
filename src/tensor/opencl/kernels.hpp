#include <CL/opencl.hpp>

#include "opencl.hpp"

#include <format>
#include <iostream>
#include <ostream>
#include <string>
#include <unordered_map>

template <typename T> class Kernels {
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

private:
  constexpr std::string getTypeName() { return "unknown"; }
  Vector vector;
  std::string configuration;

  std::string format(std::string tmp,
                     std::unordered_map<std::string, std::string> args) {
    std::string result(tmp);
    for (const auto &[key, value] : args) {
      std::string placeholder = "{" + key + "}";
      size_t pos = 0;
      while ((pos = result.find(placeholder, pos)) != std::string::npos) {
        result.replace(pos, placeholder.length(), value);
        pos += value.length();
      }
    }
    return result;
  }

  std::string unaryOperation(std::string name, std::string operation) {
    return format(
        R"(
        __kernel void {method}(__global type* A, int len) {
          int gid = get_global_id(0);
          int base = gid * WIDTH;
          if (base + WIDTH <= len) {
            typeX data = vloadX(gid, A);
            vstoreX({operation}data, gid, A);
          } else {
            for (int i = 0; i < WIDTH; i++) {
              int idx = base + i;
              if (idx < len) A[idx] = {operation}A[idx];
            }
          }
        })",
        {{"method", name}, {"operation", operation}});
  }

  std::string scalarOperation(std::string name, std::string operation) {
    return format(
        R"(
        __kernel void {method}(__global type* A, int len, type scalar) {
          int gid = get_global_id(0);
          int base = gid * WIDTH;
          if (base + WIDTH <= len) {
            typeX data = vloadX(gid, A);
            data = data {operation} scalar;
            vstoreX(data, gid, A);
          } else {
            for (int i = 0; i < WIDTH; i++) {
              int idx = base + i;
              if (idx < len) A[idx] = A[idx] {operation} scalar;
            }
          }
        })",
        {{"method", name}, {"operation", operation}});
  }

  std::string binaryOperation(std::string name, std::string operation) {
    return format(
        R"(
        __kernel void {method}(__global type* A, __global type* B, int len) {
          int gid = get_global_id(0);
          int base = gid * WIDTH;
          if (base + WIDTH <= len) {
            typeX dataA = vloadX(gid, A);
            typeX dataB = vloadX(gid, B);
            vstoreX(dataA {operation} dataB, gid, A);
          } else {
            for (int i = 0; i < WIDTH; i++) {
              int idx = base + i;
              if (idx < len) A[idx] = A[idx] {operation} B[idx];
            }
          }
        })",
        {{"method", name}, {"operation", operation}});
  }

  std::string matrixMult(std::string name) {
    return format(
        R"(
        #define TILE_SIZE WIDTH*4                      
        __kernel void mult(const __global typeX* A,
                            const __global typeX* B,
                            __global typeX* C, const int M, const int N, const int K) {
            const int row = get_local_id(0);
            const int col = get_local_id(1);
            const int globalRow = (TILE_SIZE/WIDTH)*get_group_id(0) + row;
            const int globalCol = TILE_SIZE*get_group_id(1) + col;
            __local typeX Asub[TILE_SIZE][TILE_SIZE/WIDTH];
            __local typeX Bsub[TILE_SIZE][TILE_SIZE/WIDTH];
            typeX acc = 0;
            const int numTiles = K/TILE_SIZE;
            for (int tile = 0; tile < numTiles; tile++) {
                const int tiledRow = (TILE_SIZE/WIDTH)*tile + row;
                const int tiledCol = TILE_SIZE*tile + col;
                Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];
                Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];
                barrier(CLK_LOCAL_MEM_FENCE);
                typeX vecA, vecB;
                type valB;
                for (int k = 0; k < TILE_SIZE/WIDTH; k++) {
                    vecB = Bsub[col][k];
                    for (int w = 0; w < WIDTH; w++) {
                        vecA = Asub[WIDTH*k + w][row];
                        valB = vecB[w];
                        for (int i = 0; i < WIDTH; i++)
                            acc[i] += vecA[i] * valB;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            C[globalCol*(M/WIDTH) + globalRow] = acc;
        }
        )",
        {{"method", name}});
  }

  std::unordered_map<Method, std::tuple<std::string, std::string>> programs = {
      {Method::POSITIVE, {unaryOperation("positive", "+"), "positive"}},
      {Method::NEGATIVE, {unaryOperation("negative", "-"), "negative"}},

      {Method::S_ADD, {scalarOperation("add", "+"), "add"}},
      {Method::S_MULT, {scalarOperation("mult", "*"), "mult"}},

      {Method::T_ADD, {binaryOperation("add", "+"), "add"}},
      {Method::T_HADAMARD,
       {binaryOperation("hadamard_mult", "*"), "hadamard_mult"}},

      {Method::T_MULT, {matrixMult("mult"), "mult"}},
  };

  std::unordered_map<Method, cl::Program> compiledPrograms;

public:
  Kernels(Vector vec = Vector::type4) : vector(vec) {
    std::string extensions = openCL.getDevice().getInfo<CL_DEVICE_EXTENSIONS>();
    if (extensions.find("cl_khr_fp16") != std::string::npos)
      configuration = R"(
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
        typedef half _half;
        typedef half2 _half2;
        typedef half4 _half4;
        typedef half8 _half8;
        typedef half16 _half16;
      )";
    else
      configuration = R"(
        typedef float _half;
        typedef float2 _half2;
        typedef float4 _half4;
        typedef float8 _half8;
        typedef float16 _half16;
      )";
    configuration += format(
        R"(
        typedef {type} type;
        typedef {type}{vector} typeX;
        #define WIDTH {vector}
        #define vloadX vload{vector}
        #define vstoreX vstore{vector}
      )",
        {{"type", getTypeName()}, {"vector", std::to_string((int)vector)}});

    for (const auto &[method, programInfo] : programs) {
      const auto &[sourceCode, kernelName] = programInfo;
      if (!sourceCode.empty()) {
        cl::Program program(openCL.getContext(), configuration + sourceCode);
        try {
          program.build({openCL.getDevice()});
          compiledPrograms[method] = program;
        } catch (const cl::Error &e) {
          std::cerr << "OpenCL compilation error for method "
                    << static_cast<int>(method) << ": " << e.what()
                    << std::endl;
          std::string buildLog =
              program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openCL.getDevice());
          std::cerr << "Build log for method " << static_cast<int>(method)
                    << ":" << std::endl;
          std::cerr << buildLog << std::endl;
        }
      }
    }
  }

  cl::Kernel create(Method method) {
    auto it = compiledPrograms.find(method);
    if (it == compiledPrograms.end())
      throw std::runtime_error("Program for method not found or not compiled");
    const auto &kernelName = std::get<1>(programs[method]);
    return cl::Kernel(it->second, kernelName.c_str());
  }
};

#define SPECIALIZE_KERNELS_TYPE(type, name)                                    \
  template <> constexpr std::string Kernels<type>::getTypeName() {             \
    return name;                                                               \
  }
SPECIALIZE_KERNELS_TYPE(char, "char")
SPECIALIZE_KERNELS_TYPE(short, "short")
SPECIALIZE_KERNELS_TYPE(int, "int")
SPECIALIZE_KERNELS_TYPE(long, "long")
SPECIALIZE_KERNELS_TYPE(float, "float")
SPECIALIZE_KERNELS_TYPE(double, "double")

typedef cl_half half;
SPECIALIZE_KERNELS_TYPE(half, "_half")