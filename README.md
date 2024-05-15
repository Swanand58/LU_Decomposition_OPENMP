# LU Decomposition with OpenMP

This project implements LU Decomposition using Gaussian elimination with row pivoting in a shared-memory parallel program using OpenMP.

## Overview

LU Decomposition transforms a square matrix \(A\) into the product of a lower-triangular matrix \(L\) and an upper-triangular matrix \(U\):

\[ A = LU \]

In this implementation, we perform LU decomposition with row pivoting to enhance numerical stability. The lower-triangular matrix \(L\) contains the multipliers used during Gaussian elimination, and the upper-triangular matrix \(U\) is the result of applying Gaussian elimination to matrix \(A\).

## Features

- **Parallel Implementation**: Uses OpenMP for parallelizing the LU decomposition process.
- **Row Pivoting**: Implements row pivoting to reduce round-off errors.
- **Permutation Matrix**: Computes a permutation matrix \(P\) such that \(PA = LU\).
- **Performance Measurement**: Times the LU decomposition phase and computes the L2,1 norm of the residual \(PA - LU\).

## Usage

### Compilation

Ensure you have the Intel compilers and OpenMP support. Compile the code using the provided Makefile:

```sh
make
```

## Running the Program

To run the program, execute the following command:

```sh./lu_decomposition <matrix_size> <num_threads>

```

The program takes two arguments:

- `matrix_size`: The size of the matrix to be decomposed.
- `num_threads`: The number of threads to use for parallelization.

For example, to decompose a 8000x8000 matrix with 8 threads, run the following command:

```sh
./lu_decomposition 8000 8
```

## Output

The program will output the following information:

- `LU Decomposition Time`: The time taken to perform the LU decomposition.
- `Residual L2 Norm`: The L2 norm of the residual matrix \(PA - LU\).

## Performance Evaluation

Evaluate the performance using different numbers of threads and measure the parallel efficiency. Use problem sizes of 7000 or 8000 as specified. The performance results should be tabulated and plotted for 1, 2, 4, 8, 16, and 32 threads.

## Project Report

The project report includes a detailed explanation of the implementation, parallelization strategy, performance evaluation, and results.

## References

- MathWorld LU Decomposition
- OpenMP Documentation

## Acknowledgments

Special thanks to the instructors and peers in COMP 534 for their guidance and support throughout the project.
