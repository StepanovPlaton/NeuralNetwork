#include "./math/math.hpp"

#include <chrono>
#include <thread>

using namespace GPU;

OpenCL openCL;

int main() {
  MatrixMath mm;

  Matrix a(2, 2);
  Matrix b(2, 2);

  CPU::Matrix a_(2, 2, a.toVector());
  CPU::Matrix b_(2, 2, b.toVector());

  a_.print();
  b_.print();

  Matrix c = mm.add(a, b);

  CPU::Matrix c_(2, 2, c.toVector(&mm.getQueue()));

  mm.await();

  c_.print();

  return 0;
}