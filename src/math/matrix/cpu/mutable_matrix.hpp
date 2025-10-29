#pragma once

#include "matrix.hpp"

#include "../mutable_matrix.hpp"

#include <cmath>

#define M_PI 3.14159265358979323846

namespace MutableMatrices {

class CPU : public Matrices::CPU, public IMutableMatrix<Matrices::CPU> {
private:
  static float activate_x(float x, Activate type, float alpha = 0.01f);

public:
  CPU(int rows, int cols, const std::vector<float> &matrix)
      : Matrices::CPU(rows, cols, matrix) {}

  void mult(Matrices::CPU &m, float bias = 0.0f,
            Activate type = Activate::LINEAR, float alpha = 0.01f);
  void mult(float scalar);
  void add(Matrices::CPU &m, float a = 1.0f, float b = 1.0f);
  void add(float scalar);
  void activate(Activate type, float alpha = 0.01f);
};
}; // namespace MutableMatrices
