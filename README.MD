# Ridge Regression Parallel Benchmark

This project explores parallel techniques for training ridge regression models and benchmarks their execution times. Implemented in **C++** and **CUDA**, it leverages GPU acceleration to enhance computational efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Benchmarking](#benchmarking)
- [References](#references)
- [License](#license)

## Introduction

Ridge regression is a technique for analyzing multiple regression data that suffer from multicollinearity. By introducing a degree of bias to the regression estimates, it reduces standard errors. This project implements ridge regression using parallel computing techniques to improve training performance, utilizing both CPU (via MPI, OpenMP, Multithread) and GPU (via CUDA) parallelism.

## Features

- ✅ Parallel CPU Implementation using OpenMP
- 🚀 GPU Acceleration with CUDA
- 📊 Benchmarking suite for comparing execution times

## Prerequisites

- Operating System: Linux or Windows
- Compiler: GCC (Linux) or MSVC (Windows)
- CUDA Toolkit (for GPU support)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Thanutham-T/ridge_regression_parallel_benchmark.git
   cd ridge_regression_parallel_benchmark
   ```

2. **Prepare Data** (Using git lfs):
    ```bash
    git lfs pull
    ```

## Usage

1. **Compiles the project** (Using Makefile):

   - Compiles Normal:

        ```bash
        make compile-all-normal
        ```

    - Compiles with optimize:

        ```bash
        make compile-all-optimize
        ```

2. **Run test** (Using Makefile):

    - Compiles setting:

        ```bash
        make run-all
        ```
    
    - Custom setting for run (MPI + OpenMP):

        ```bash
        make run-all OMP_NUM_THREADS=<NUM_OF_THREAD> # default is 2
        ```

3. **Clean build files**

    ```bash
    make clean
    ```

## Project Structure

ridge_regression_parallel_benchmark/  
├── data/               # Dataset files  
├── source/             # C++ and CUDA source code  
├── makefile            # Build instructions  
├── README.md           # Project documentation

## Benchmarking

The benchmark compares performance across:  
- 🧠 **Serial CPU**: Standard ridge regression  
- 🧵 **Parallel CPU (MPI, OpenMP, Multithread)**: Multithreaded optimization  
- 💻 **GPU (CUDA)**: GPU-accelerated training  

This helps identify the most efficient implementation for large datasets.

