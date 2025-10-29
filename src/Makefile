CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++11
LIBS = -lOpenCL
TARGET = main
SRC = main.cpp ./math/opencl/opencl.cpp ./math/matrix/cpu/matrix.cpp ./math/matrix/cpu/mutable_matrix.cpp ./math/matrix/gpu/matrix.cpp ./math/matrix/gpu/mutable_matrix.cpp

INCLUDES = -I"A:/Programs/OpenCL/include" 
LIB_PATH = -L"A:/Programs/OpenCL/lib"

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_PATH) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run