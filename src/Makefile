CXX = g++
CXXFLAGS = -Wall -O2 -std=c++11
LIBS = -lOpenCL
TARGET = main
SRC = main.cpp

INCLUDES = -I"A:/Programs/OpenCL/include" 
LIB_PATH = -L"A:/Programs/OpenCL/lib"

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIB_PATH) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run