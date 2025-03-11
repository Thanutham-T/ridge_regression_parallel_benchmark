# Compiler settings
CXX = g++
MPICXX = mpic++
CXXFLAGS = -Wall -std=c++20
OPTFLAGS = -O3

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

# Output executables
OUT_NORM = $(OUTDIR)/ridge_regression
OUT_MT = $(OUTDIR)/ridge_regression_MultiThread
OUT_ASYNC = $(OUTDIR)/ridge_regression_AsyncThread
OUT_OPENMP = $(OUTDIR)/ridge_regression_OpenMP
OUT_MPI = $(OUTDIR)/ridge_regression_MPI

# Create output directory if it doesn't exist
$(shell mkdir -p $(OUTDIR))

# Normal compilation
compile-all-normal: $(OUT_NORM) $(OUT_MT) $(OUT_ASYNC) $(OUT_OPENMP) $(OUT_MPI)

$(OUT_NORM): $(SRC_NORM)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(OUT_MT): $(SRC_MT)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

$(OUT_ASYNC): $(SRC_ASYNC)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -pthread $< -o $@

$(OUT_OPENMP): $(SRC_OPENMP)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

$(OUT_MPI): $(SRC_MPI)
	@mkdir -p $(OUTDIR)
	$(MPICXX) $(CXXFLAGS) $< -o $@

# Optimized compilation (-O3)
compile-all-optimize: $(OUT_NORM)_opt $(OUT_MT)_opt $(OUT_ASYNC)_opt $(OUT_OPENMP)_opt $(OUT_MPI)_opt

$(OUT_NORM)_opt: $(SRC_NORM)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $< -o $(OUT_NORM)

$(OUT_MT)_opt: $(SRC_MT)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -pthread $< -o $(OUT_MT)

$(OUT_ASYNC)_opt: $(SRC_ASYNC)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -pthread $< -o $(OUT_ASYNC)

$(OUT_OPENMP)_opt: $(SRC_OPENMP)
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fopenmp $< -o $(OUT_OPENMP)

$(OUT_MPI)_opt: $(SRC_MPI)
	@mkdir -p $(OUTDIR)
	$(MPICXX) $(CXXFLAGS) $(OPTFLAGS) $< -o $(OUT_MPI)

# Run all executables
run-all:
	@echo "Running Sequential..."
	@$(OUT_NORM)
	@echo "\n"

	@echo "Running MultiThread..."
	@$(OUT_MT)
	@echo "\n"

	@echo "Running AsyncThread..."
	@$(OUT_ASYNC)
	@echo "\n"

	@echo "Running OpenMP..."
	@$(OUT_OPENMP)
	@echo "\n"

	@echo "Running MPI..."
	@mpirun -np 5 $(OUT_MPI)
	@echo "\n"

# Clean build files
clean:
	rm -rf $(OUTDIR)/*
