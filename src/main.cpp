#include <CL/cl.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "device.hpp"
#include "matrix.hpp"

class MatrixCalculator {
private:
  CalcEngine *calcEngine;
  cl_command_queue queue;
  cl_kernel kernel;

public:
  MatrixCalculator(CalcEngine &calcEngine) {
    this->calcEngine = &calcEngine;
    kernel = calcEngine.loadKernel("matrix_mult.cl");
    queue = clCreateCommandQueue(calcEngine.getContext(),
                                 calcEngine.getDevice(), 0, nullptr);
    if (!queue) {
      throw OpenCLException(-1, "clCreateCommandQueue");
    }
  }

  ~MatrixCalculator() {
    if (queue)
      clReleaseCommandQueue(queue);
  }

  std::vector<float> multiply(Matrix &a, Matrix &b, int M, int N, int K) {
    if (a.getRows() != M || a.getCols() != K || b.getRows() != K ||
        b.getCols() != N) {
      throw std::invalid_argument("Invalid matrix dimensions");
    }

    cl_mem bufC = calcEngine->createBuffer(CL_MEM_WRITE_ONLY,
                                           M * N * sizeof(float), nullptr);

    calcEngine->setKernelArgs(kernel, a.getBuf(), b.getBuf(), bufC, M, N, K);

    calcEngine->runKernel(queue, kernel, M, N);

    std::vector<float> C(M * N);
    calcEngine->readResult(queue, bufC, C);

    clReleaseMemObject(bufC);

    return C;
  }
};

int main() {
  CalcEngine calcEngine;
  calcEngine.printDeviceInfo();

  MatrixCalculator matrixCalculator(calcEngine);

  float matrixA[2 * 3] = {1, 2, 3, 4, 5, 6};
  Matrix a(calcEngine, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2, 3, matrixA);

  float matrixB[3 * 2] = {1, 2, 3, 4, 5, 6};
  Matrix b(calcEngine, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 3, 2, matrixB);

  std::vector<float> v = matrixCalculator.multiply(a, b, 2, 2, 3);
  for (const auto &element : v) {
    std::cout << element << " ";
  }

  return 0;
}