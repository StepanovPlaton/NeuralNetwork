#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "./math/math.hpp"

typedef Matrices::CPU Matrix;
typedef MutableMatrices::CPU MutableMatrix;

OpenCL openCL;

std::vector<float> generateRandomMatrix(int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<float> matrix(rows * cols);
  for (int i = 0; i < rows * cols; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}
std::vector<float> generateIdentityMatrix(int size) {
  std::vector<float> matrix(size * size, 0.0f);
  for (int i = 0; i < size; ++i) {
    matrix[i * size + i] = 1.0f;
  }
  return matrix;
}

int main() {
  const int SIZE = 1024;

  std::cout << "Testing with " << SIZE << "x" << SIZE << " matrices..."
            << std::endl;

  std::vector<float> matrixA = generateRandomMatrix(SIZE, SIZE);
  std::vector<float> matrixB = generateRandomMatrix(SIZE, SIZE);
  std::vector<float> matrixC = generateRandomMatrix(SIZE, SIZE);

  // std::vector<float> matrixA = generateIdentityMatrix(SIZE);
  // std::vector<float> matrixB = generateIdentityMatrix(SIZE);
  // std::vector<float> matrixC = generateIdentityMatrix(SIZE);

  // Тестирование на CPU
  {
    std::cout << "\n=== CPU Version ===" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    MutableMatrices::CPU a(SIZE, SIZE, matrixA);
    Matrices::CPU b(SIZE, SIZE, matrixB);
    Matrices::CPU c(SIZE, SIZE, matrixC);

    auto gen_end = std::chrono::high_resolution_clock::now();

    auto op_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; i++) {
      a.mult(b);
    }

    auto op_end = std::chrono::high_resolution_clock::now();

    std::vector<float> v = a.toVector();

    auto total_end = std::chrono::high_resolution_clock::now();

    auto gen_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - start);
    auto op_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        op_end - op_start);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - start);

    std::cout << "Matrix generation time: " << gen_duration.count() << " ms"
              << std::endl;
    std::cout << "Operations time: " << op_duration.count() << " ms"
              << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;

    std::cout << "First few elements: ";
    for (int i = 0; i < 5 && i < v.size(); ++i) {
      std::cout << v[i] << " ";
    }
    std::cout << std::endl;
  }

  // Тестирование на GPU
  {
    std::cout << "\n=== GPU Version ===" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    MutableMatrices::GPU a(SIZE, SIZE, matrixA);
    Matrices::GPU b(SIZE, SIZE, matrixB);
    Matrices::GPU c(SIZE, SIZE, matrixC);

    auto gen_end = std::chrono::high_resolution_clock::now();

    auto op_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; i++) {
      a.mult(b);
    }

    auto op_end = std::chrono::high_resolution_clock::now();

    std::vector<float> v = a.toVector();

    auto total_end = std::chrono::high_resolution_clock::now();

    auto gen_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - start);
    auto op_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        op_end - op_start);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - start);

    std::cout << "Matrix generation time: " << gen_duration.count() << " ms"
              << std::endl;
    std::cout << "Operations time: " << op_duration.count() << " ms"
              << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;

    std::cout << "First few elements: ";
    for (int i = 0; i < 5 && i < v.size(); ++i) {
      std::cout << v[i] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}