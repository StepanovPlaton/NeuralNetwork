#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "./math/math.hpp"

using namespace GPU;

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
  const int SIZE = 48;

  std::cout << "Testing with " << SIZE << "x" << SIZE << " matrices..."
            << std::endl;

  // std::vector<float> matrixA = generateRandomMatrix(SIZE, SIZE);
  // std::vector<float> matrixB = generateRandomMatrix(SIZE, SIZE);
  // std::vector<float> matrixC = generateRandomMatrix(SIZE, SIZE);

  std::vector<float> matrixA = generateIdentityMatrix(SIZE);
  std::vector<float> matrixB = generateIdentityMatrix(SIZE);
  std::vector<float> matrixC = generateIdentityMatrix(SIZE);

  // Тестирование на GPU
  {
    std::cout << "\n=== GPU Version ===" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    MatrixMath mm;
    Matrix a(SIZE, SIZE, matrixA);
    Matrix b(SIZE, SIZE, matrixB);

    auto gen_end = std::chrono::high_resolution_clock::now();
    auto op_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
      Matrix x = mm.mult(a, b);
    }
    auto op_end = std::chrono::high_resolution_clock::now();

    std::vector<float> v = a.toVector(&mm.getQueue());

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
    for (size_t i = 0; i < 5 && i < v.size(); ++i) {
      std::cout << v[i] << " ";
    }
    std::cout << std::endl;
  }

  // Тестирование на CPU
  {
    std::cout << "\n=== CPU Version ===" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    CPU::MatrixMath mm;
    CPU::Matrix a(SIZE, SIZE, matrixA);
    CPU::Matrix b(SIZE, SIZE, matrixB);

    auto gen_end = std::chrono::high_resolution_clock::now();

    auto op_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
      CPU::Matrix x = mm.mult(a, b);
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
    for (size_t i = 0; i < 5 && i < v.size(); ++i) {
      std::cout << v[i] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}