#pragma once

#include "../../opencl/opencl.hpp"

#include "matrix.hpp"

#include "../mutable_matrix.hpp"

namespace MutableMatrices {

class GPU : public Matrices::GPU, public IMutableMatrix<Matrices::GPU> {
private:
  enum class Method { MULT, SCALAR_MULT, ADD, SCALAR_ADD, ACTIVATE };
  std::unordered_map<Method, cl::Kernel> kernels;
  std::unordered_map<Method, std::string> kernelsNames = {
      {Method::MULT, "mult"},
      {Method::SCALAR_MULT, "mult_sc"},
      {Method::ADD, "add"},
      {Method::SCALAR_ADD, "add_sc"},
      {Method::ACTIVATE, "activate"}};

  static void CL_CALLBACK releaseBuffer(cl_event, cl_int status, void *buf) {
    if (status == CL_COMPLETE) {
      //   std::cout << "Kernel complete!" << std::endl;
      delete (cl::Buffer *)buf;
    }
  }

public:
  GPU(int rows, int cols, const std::vector<float> &matrix);

  void mult(Matrices::GPU &m, float bias = 0.0f,
            Activate type = Activate::LINEAR, float alpha = 0.01f);
  void mult(float scalar);
  void add(Matrices::GPU &m, float a = 1.0f, float b = 1.0f);
  void add(float scalar);
  void activate(Activate type, float alpha = 0.01f);
};

}; // namespace MutableMatrices
