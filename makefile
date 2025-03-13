# Compiler settings
CXX = g++
MPICXX = mpic++
CXXFLAGS = -Wall -std=c++20
OPTFLAGS = -O3
NVCC = nvcc
CUDAOPTFLAGS = -gencode arch=compute_75,code=sm_75

# SOURCE directory
SOURCE = ./source

# Output directory
OUTDIR = ./out

# Source files
SRC_NORM = $(SOURCE)/ridge_regression.cpp
SRC_MT = $(SOURCE)/ridge_regression_MultiThread.cpp
SRC_ASYNC = $(SOURCE)/ridge_regression_AsyncThread.cpp
SRC_OPENMP = $(SOURCE)/ridge_regression_OpenMP.cpp
SRC_MPI = $(SOURCE)/ridge_regression_MPI.cpp
SRC_OMPI = $(SOURCE)/ridge_regression_OMPI.cpp
SRC_CUDA = $(SOURCE)/ridge_regression_cuda.cu

# Output executables
OUT_NORM = $(OUTDIR)/ridge_regression
OUT_MT = $(OUTDIR)/ridge_regression_MultiThread
OUT_ASYNC = $(OUTDIR)/ridge_regression_AsyncThread
OUT_OPENMP = $(OUTDIR)/ridge_regression_OpenMP
OUT_MPI = $(OUTDIR)/ridge_regression_MPI
OUT_OMPI = $(OUTDIR)/ridge_regression_OMPI
OUT_CUDA = $(OUTDIR)/ridge_regression_cuda

# Create output directory if it doesn't exist
$(shell mkdir -p $(OUTDIR))

# Normal compilation
compile-all-normal: $(OUT_NORM) $(OUT_MT) $(OUT_ASYNC) $(OUT_OPENMP) $(OUT_MPI)

$(OUT_NORM): $(SRC_NORM)
	$(CXX) $(CXXFLAGS) $< -o $@

$(OUT_MT): $(SRC_MT)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

$(OUT_ASYNC): $(SRC_ASYNC)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

$(OUT_OPENMP): $(SRC_OPENMP)
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

$(OUT_MPI): $(SRC_MPI)
	$(MPICXX) $(CXXFLAGS) $< -o $@

$(OUT_OMPI): $(SRC_OMPI)
	$(MPICXX) $(CXXFLAGS) -fopenmp $< -o $@

$(OUT_CUDA): $(SRC_CUDA)
	$(NVCC) $< -o $@

# Optimized compilation (-O3)
compile-all-optimize: 
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(SRC_NORM) -o $(OUT_NORM)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -pthread $(SRC_MT) -o $(OUT_MT)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -pthread $(SRC_ASYNC) -o $(OUT_ASYNC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fopenmp $(SRC_OPENMP) -o $(OUT_OPENMP)
	$(MPICXX) $(CXXFLAGS) $(OPTFLAGS) $(SRC_MPI) -o $(OUT_MPI)
	$(MPICXX) $(CXXFLAGS) $(OPTFLAGS) -fopenmp $(SRC_OMPI) -o $(OUT_OMPI)
	$(NVCC)  $(OPTFLAGS) $(CUDAOPTFLAGS) $(SRC_CUDA) -o $(OUT_CUDA)

OMP_NUM_THREADS?=2

run-all:
	@echo "Running Normal..."
	@$(OUTDIR)/ridge_regression
	@echo "\n"

	@echo "Running MultiThread..."
	@$(OUTDIR)/ridge_regression_MultiThread
	@echo "\n"

	@echo "Running AsyncThread..."
	@$(OUTDIR)/ridge_regression_AsyncThread
	@echo "\n"

	@echo "Running OpenMP..."
	@$(OUTDIR)/ridge_regression_OpenMP
	@echo "\n"

	@echo "Running MPI..."
	@mpirun -np 5 $(OUTDIR)/ridge_regression_MPI
	@echo "\n"

	@echo "Running MPI & OpenMP..."
	@export OMP_NUM_THREADS=$(OMP_NUM_THREADS)
	@mpirun -np 3 $(OUTDIR)/ridge_regression_OMPI
	@echo "\n"

	@echo "Running CUDA..."
	@$(OUTDIR)/ridge_regression_cuda
	@echo "\n"

# Clean build files
clean:
	rm -rf $(OUTDIR)/*
