CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++23
LIBS = -lOpenCL
TARGET = main
COMMON_SRC = ./math/opencl/opencl.cpp ./math/matrix/cpu/matrix.cpp ./math/matrix/cpu/mutable_matrix.cpp ./math/matrix/gpu/matrix.cpp ./math/matrix/gpu/mutable_matrix.cpp
MAIN_SRC = main.cpp $(COMMON_SRC)
BENCHMARK_SRC = benchmark.cpp $(COMMON_SRC)

INCLUDES = -I"A:/Programs/OpenCL/include" 
LIB_PATH = -L"A:/Programs/OpenCL/lib"

$(TARGET): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_PATH) -o $(TARGET) $(MAIN_SRC) $(LIBS)

benchmark: $(BENCHMARK_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_PATH) -o $(TARGET) $(BENCHMARK_SRC) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

run_benchmark: benchmark
	./$(TARGET)

.PHONY: clean run