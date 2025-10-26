#include <CL/cl.h>

#include <stdexcept>
#include <vector>

#include "device.hpp"
#include "matrix.hpp"

class MutableMatrix : public Matrix {
private:
  CalcEngine *calcEngine;
  cl_command_queue queue;
  cl_kernel kernel;

public:
  MutableMatrix(CalcEngine &calcEngine, size_t rows, size_t cols, float *matrix)
      : Matrix(calcEngine, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rows, cols,
               matrix) {
    this->calcEngine = &calcEngine;
    kernel = calcEngine.loadKernel("matrix_mult.cl");
    queue = clCreateCommandQueue(calcEngine.getContext(),
                                 calcEngine.getDevice(), 0, nullptr);
    if (!queue) {
      throw OpenCLException(-1, "clCreateCommandQueue");
    }
  }

  ~MutableMatrix() {
    if (queue)
      clReleaseCommandQueue(queue);
  }

  void mult_by(Matrix &m) {
    if (cols != m.getRows()) {
      throw std::invalid_argument("Invalid matrix dimensions");
    }

    cl_mem b =
        calcEngine->createBuffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 rows * m.getCols() * sizeof(float), nullptr);

    calcEngine->setKernelArgs(kernel, buf, m.getBuf(), b, rows, m.getCols(),
                              cols);
    calcEngine->runKernel(queue, kernel, rows, m.getCols());

    clReleaseMemObject(buf);
    buf = b;
  }

  std::vector<float> exportMatrix() {
    std::vector<float> C(rows, cols);
    calcEngine->readResult(queue, buf, C);
    return C;
  }
};

int main() {
  CalcEngine calcEngine;
  calcEngine.printDeviceInfo();

  float matrixA[2 * 3] = {1, 2, 3, 4, 5, 6};
  MutableMatrix a(calcEngine, 2, 3, matrixA);

  float matrixB[3 * 2] = {1, 2, 3, 4, 5, 6};
  Matrix b(calcEngine, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 3, 2, matrixB);

  a.mult_by(b);

  std::vector<float> v = a.exportMatrix();
  for (const auto &element : v) {
    std::cout << element << " ";
  }

  return 0;
}
