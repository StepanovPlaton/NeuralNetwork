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
COMMON_SRC = 
OPENCL_SRC = opencl/opencl.cpp

PYTHON_PATH = $(shell python -c "from sysconfig import get_paths; print(get_paths()['data'])")
PYTHON_INCLUDE = $(shell python -c "import sysconfig; print(sysconfig.get_config_var([k for k in sysconfig.get_config_vars().keys() if 'INCLUDE' in k][0]))")
PYTHON_LIB_PATH = $(PYTHON_PATH)$(SP)libs
PYTHON_LIB = -lpython313 #-lpython3.13

PYBIND_INCLUDE = $(shell python -c "import pybind11; print(pybind11.get_include())")

OPENCL_INCLUDES = -I"A:/Programs/OpenCL/include" 
OPENCL_LIB_PATH = -L"A:/Programs/OpenCL/lib"
OPENCL_LIB = -lOpenCL

.DEFAULT_GOAL := cpu

$(BUILD_DIR):
	$(MKDIR) $(BUILD_DIR)

cpu: $(COMMON_SRC) main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CPU -o $(TARGET) $^ 

opencl: $(COMMON_SRC) $(OPENCL_SRC) main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_OPENCL $(OPENCL_INCLUDES) $(OPENCL_LIB_PATH) -o $(TARGET) $^ $(OPENCL_LIB)

cpu_module: $(COMMON_SRC) pybind.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CPU -shared -fPIC -I"$(PYTHON_INCLUDE)" -I"$(PYBIND_INCLUDE)" -L"$(PYTHON_LIB_PATH)" -o tensor.$(SHARED_LIB_EXT) $^ $(PYTHON_LIB)
	PYTHONPATH=. pybind11-stubgen tensor -o .

opencl_module: $(COMMON_SRC) $(OPENCL_SRC) pybind.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_OPENCL -shared -fPIC -I"$(PYTHON_INCLUDE)" -I"$(PYBIND_INCLUDE)" -L"$(PYTHON_LIB_PATH)" $(OPENCL_INCLUDES) $(OPENCL_LIB_PATH) -o tensor.$(SHARED_LIB_EXT) $^ $(PYTHON_LIB) $(OPENCL_LIB)
	PYTHONPATH=. pybind11-stubgen tensor -o .

clean:
	rm -rf $(BUILD_DIR) $(TARGET) *.$(SHARED_LIB_EXT)
