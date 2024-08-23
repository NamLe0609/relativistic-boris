# Compiler and flags
NVCC = nvcc
CXX = g++
EXECUTABLE = run.exe

CUDA_PATH = /usr/local/cuda/
NVCCFLAGS = -O3 
CXXFLAGS = -O3 -I $(CUDA_PATH)/include -Iinclude
LDFLAGS = -L $(CUDA_PATH)/lib64 -lcudart

# Directories
SRC_DIR = src
CPP_DIR = $(SRC_DIR)/cpp
CUDA_DIR = $(SRC_DIR)/cuda
INCLUDE_DIR = include
OBJECT_DIR = obj
BIN_DIR = bin

# Source files
CPP_SRC = $(wildcard $(CPP_DIR)/*.cpp)
CUDA_SRC = $(wildcard $(CUDA_DIR)/*.cu)

# Object files
CPP_OBJ = $(CPP_SRC:$(CPP_DIR)/%.cpp=$(OBJECT_DIR)/%.o)
CUDA_OBJ = $(CUDA_SRC:$(CUDA_DIR)/%.cu=$(OBJECT_DIR)/%.o)
OBJ = $(CPP_OBJ) $(CUDA_OBJ)

# Executable name
TARGET = $(BIN_DIR)/$(EXECUTABLE)

# Targets and rules
all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $(OBJ)

# Rule to compile CUDA source files
$(OBJECT_DIR)/%.o: $(CUDA_DIR)/%.cu
	@mkdir -p $(OBJECT_DIR)
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDE_DIR) -c $< -o $@

# Rule to compile C++ source files
$(OBJECT_DIR)/%.o: $(CPP_DIR)/%.cpp
	@mkdir -p $(OBJECT_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -I $(INCLUDE_DIR) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJECT_DIR)/*.o $(TARGET)
	@rmdir $(OBJECT_DIR) 2>/dev/null || true
	@rmdir $(BIN_DIR) 2>/dev/null || true

.PHONY: all clean
