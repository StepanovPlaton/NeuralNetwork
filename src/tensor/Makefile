CXX = g++
CXXFLAGS = -Wall -Wextra -Wpedantic -O1 -g -std=c++23

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif
ifeq ($(DETECTED_OS),Windows)
	TARGET = main.exe
    MKDIR = powershell -Command "mkdir"
    SHARED_LIB_EXT = pyd
	SP = \\
else
	TARGET = main
    MKDIR = mkdir -p
    SHARED_LIB_EXT = so
	SP = /
endif

BUILD_DIR = build
COMMON_SRC = opencl/opencl.cpp

PYTHON_PATH = $(shell python -c "from sysconfig import get_paths; print(get_paths()['data'])")
PYTHON_INCLUDE = $(shell python -c "import sysconfig; print(sysconfig.get_config_var('CONFINCLUDEPY'))")
PYTHON_LIBS = $(PYTHON_PATH)$(SP)libs
PYBIND_INCLUDE = $(shell python -c "import pybind11; print(pybind11.get_include())")

OPENCL_INCLUDES = -I"A:/Programs/OpenCL/include" 
OPENCL_LIB_PATH = -L"A:/Programs/OpenCL/lib" -lOpenCL
OPENCL_LIB = -L"A:/Programs/OpenCL/lib" -lOpenCL

.DEFAULT_GOAL := $(TARGET)

$(BUILD_DIR):
	$(MKDIR) $(BUILD_DIR)

$(TARGET): $(COMMON_SRC) main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCL_INCLUDES) $(OPENCL_LIB_PATH) -o $@ $^ $(OPENCL_LIB)

module: $(COMMON_SRC) pybind.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o tensor.$(SHARED_LIB_EXT) $^ -I"$(PYTHON_INCLUDE)" -L"$(PYTHON_LIBS)" -lpython3.13 -I"$(PYBIND_INCLUDE)"
	PYTHONPATH=. pybind11-stubgen tensor -o .

clean:
	rm -rf $(BUILD_DIR) $(TARGET) *.$(SHARED_LIB_EXT)
