#include <CL/cl.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Чтение файла в строку
std::string readFile(const char *filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(std::string("Failed to open file: ") + filename);
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

// Получение ошибки OpenCL в виде строки
const char *getErrorString(cl_int error) {
  switch (error) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE_TYPE:
    return "CL_INVALID_DEVICE_TYPE";
  case CL_INVALID_PLATFORM:
    return "CL_INVALID_PLATFORM";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_QUEUE_PROPERTIES:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_HOST_PTR:
    return "CL_INVALID_HOST_PTR";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CL_INVALID_IMAGE_SIZE:
    return "CL_INVALID_IMAGE_SIZE";
  case CL_INVALID_SAMPLER:
    return "CL_INVALID_SAMPLER";
  case CL_INVALID_BINARY:
    return "CL_INVALID_BINARY";
  case CL_INVALID_BUILD_OPTIONS:
    return "CL_INVALID_BUILD_OPTIONS";
  case CL_INVALID_PROGRAM:
    return "CL_INVALID_PROGRAM";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case CL_INVALID_KERNEL_NAME:
    return "CL_INVALID_KERNEL_NAME";
  case CL_INVALID_KERNEL_DEFINITION:
    return "CL_INVALID_KERNEL_DEFINITION";
  case CL_INVALID_KERNEL:
    return "CL_INVALID_KERNEL";
  case CL_INVALID_ARG_INDEX:
    return "CL_INVALID_ARG_INDEX";
  case CL_INVALID_ARG_VALUE:
    return "CL_INVALID_ARG_VALUE";
  case CL_INVALID_ARG_SIZE:
    return "CL_INVALID_ARG_SIZE";
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_DIMENSION:
    return "CL_INVALID_WORK_DIMENSION";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case CL_INVALID_GLOBAL_OFFSET:
    return "CL_INVALID_GLOBAL_OFFSET";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_INVALID_EVENT:
    return "CL_INVALID_EVENT";
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case CL_INVALID_GL_OBJECT:
    return "CL_INVALID_GL_OBJECT";
  case CL_INVALID_BUFFER_SIZE:
    return "CL_INVALID_BUFFER_SIZE";
  case CL_INVALID_MIP_LEVEL:
    return "CL_INVALID_MIP_LEVEL";
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  default:
    return "Unknown OpenCL error";
  }
}

// Проверка ошибок OpenCL
void checkError(cl_int err, const char *operation) {
  if (err != CL_SUCCESS) {
    std::cerr << "Error during " << operation << ": " << getErrorString(err)
              << " (" << err << ")" << std::endl;
    exit(1);
  }
}

// Код ядра для матричного умножения с тайлингом
const char *kernelSource = R"(
__kernel void matmul_tiled(__global const float* A,
                          __global const float* B,
                          __global float* C,
                          const int N,
                          const int TILE_SIZE) {
    
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    __local float tileA[16][16];
    __local float tileB[16][16];
    
    float sum = 0.0f;
    
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Загрузка тайлов в локальную память
        int tileRow = get_local_id(1);
        int tileCol = get_local_id(0);
        
        int loadRow = row;
        int loadCol = t * TILE_SIZE + tileCol;
        if (loadRow < N && loadCol < N) {
            tileA[tileRow][tileCol] = A[loadRow * N + loadCol];
        } else {
            tileA[tileRow][tileCol] = 0.0f;
        }
        
        loadRow = t * TILE_SIZE + tileRow;
        loadCol = col;
        if (loadRow < N && loadCol < N) {
            tileB[tileRow][tileCol] = B[loadRow * N + loadCol];
        } else {
            tileB[tileRow][tileCol] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Вычисление частичной суммы
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tileRow][k] * tileB[k][tileCol];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
)";

int main() {
  cl_int err;

  // Параметры матрицы
  const int N = 1024; // Размер матрицы (уменьшено для демонстрации)
  const int TILE_SIZE = 16;
  const size_t matrixSize = N * N * sizeof(float);

  std::cout << "Matrix size: " << N << "x" << N << " (" << N * N << " elements)"
            << std::endl;
  std::cout << "Total data: " << matrixSize / (1024 * 1024) << " MB per matrix"
            << std::endl;

  // Инициализация данных
  std::vector<float> A(N * N);
  std::vector<float> B(N * N);
  std::vector<float> C(N * N, 0.0f);

  // Заполнение матриц тестовыми данными
  for (int i = 0; i < N * N; i++) {
    A[i] = static_cast<float>(i % 100) * 0.1f;
    B[i] = static_cast<float>((i + 1) % 100) * 0.1f;
  }

  // 1. Получение платформы
  cl_platform_id platform;
  err = clGetPlatformIDs(1, &platform, NULL);
  checkError(err, "clGetPlatformIDs");

  // 2. Получение устройства (GPU)
  cl_device_id device;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    std::cout << "GPU not found, trying CPU..." << std::endl;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    checkError(err, "clGetDeviceIDs");
    std::cout << "Using CPU" << std::endl;
  } else {
    std::cout << "Using GPU" << std::endl;
  }

  // 3. Создание контекста
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  checkError(err, "clCreateContext");

  // 4. Создание очереди команд
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  checkError(err, "clCreateCommandQueue");

  // 5. Создание буферов
  cl_mem bufferA =
      clCreateBuffer(context, CL_MEM_READ_ONLY, matrixSize, NULL, &err);
  checkError(err, "clCreateBuffer A");

  cl_mem bufferB =
      clCreateBuffer(context, CL_MEM_READ_ONLY, matrixSize, NULL, &err);
  checkError(err, "clCreateBuffer B");

  cl_mem bufferC =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrixSize, NULL, &err);
  checkError(err, "clCreateBuffer C");

  // 6. Копирование данных на устройство
  auto copy_start = std::chrono::high_resolution_clock::now();

  err = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, matrixSize, A.data(),
                             0, NULL, NULL);
  checkError(err, "clEnqueueWriteBuffer A");

  err = clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, matrixSize, B.data(),
                             0, NULL, NULL);
  checkError(err, "clEnqueueWriteBuffer B");

  auto copy_end = std::chrono::high_resolution_clock::now();
  auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(
      copy_end - copy_start);

  // 7. Создание программы
  auto program_start = std::chrono::high_resolution_clock::now();

  cl_program program =
      clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
  checkError(err, "clCreateProgramWithSource");

  // Компиляция программы
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    // Получение логов компиляции
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                          log.data(), NULL);
    std::cerr << "Build failed:\n" << log.data() << std::endl;
    checkError(err, "clBuildProgram");
  }

  auto program_end = std::chrono::high_resolution_clock::now();
  auto program_time = std::chrono::duration_cast<std::chrono::microseconds>(
      program_end - program_start);

  // 8. Создание ядра
  auto kernel_start = std::chrono::high_resolution_clock::now();

  cl_kernel kernel = clCreateKernel(program, "matmul_tiled", &err);
  checkError(err, "clCreateKernel");

  auto kernel_end = std::chrono::high_resolution_clock::now();
  auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(
      kernel_end - kernel_start);

  // 9. Установка аргументов ядра
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
  checkError(err, "clSetKernelArg 0");

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
  checkError(err, "clSetKernelArg 1");

  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
  checkError(err, "clSetKernelArg 2");

  err = clSetKernelArg(kernel, 3, sizeof(int), &N);
  checkError(err, "clSetKernelArg 3");

  err = clSetKernelArg(kernel, 4, sizeof(int), &TILE_SIZE);
  checkError(err, "clSetKernelArg 4");

  // 10. Запуск матричного умножения
  size_t global[2] = {N, N};
  size_t local[2] = {TILE_SIZE, TILE_SIZE};

  auto matmul_start = std::chrono::high_resolution_clock::now();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL,
                               NULL);
  checkError(err, "clEnqueueNDRangeKernel");

  clFinish(queue);

  auto matmul_end = std::chrono::high_resolution_clock::now();
  auto matmul_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      matmul_end - matmul_start);

  // 11. Чтение результатов
  auto read_start = std::chrono::high_resolution_clock::now();

  err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, matrixSize, C.data(), 0,
                            NULL, NULL);
  checkError(err, "clEnqueueReadBuffer");

  auto read_end = std::chrono::high_resolution_clock::now();
  auto read_time = std::chrono::duration_cast<std::chrono::microseconds>(
      read_end - read_start);

  // Вывод результатов измерений
  std::cout << "\n=== TIMING RESULTS ===" << std::endl;
  std::cout << "Data copy to device: " << copy_time.count() << " ns"
            << std::endl;
  std::cout << "Program creation: " << program_time.count() << " ns"
            << std::endl;
  std::cout << "Kernel creation: " << kernel_time.count() << " ns" << std::endl;
  std::cout << "Matrix multiplication: " << matmul_time.count() << " ms"
            << std::endl;
  std::cout << "Data read from device: " << read_time.count() << " ns"
            << std::endl;

  // Расчет отношения времени выполнения к времени создания ядра
  if (kernel_time.count() > 0) {
    long long ratio = (matmul_time.count() * 1000) /
                      kernel_time.count(); // переводим ms в ns для сравнения
    std::cout << "Kernel creation vs execution ratio: 1 : " << ratio
              << std::endl;
  }

  // Расчет производительности
  long long total_flops = 2LL * N * N * N; // 2*N^3 FLOP
  double gflops = (double)total_flops / (matmul_time.count() * 1e6); // GFLOP/s
  std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

  // Проверка результата (простая валидация)
  float checksum = 0.0f;
  for (int i = 0; i < N * N; i++) {
    checksum += C[i];
  }
  std::cout << "Result checksum: " << checksum << std::endl;

  // 12. Освобождение ресурсов
  clReleaseMemObject(bufferA);
  clReleaseMemObject(bufferB);
  clReleaseMemObject(bufferC);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  std::cout << "\nDone!" << std::endl;

  return 0;
}