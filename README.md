# Autograd in C++ - A Minimal Autograd Engine

A from scratch automatic differentiation engine in c++
Built to understand how production ready libraries like PyTorch work

## What it has
- **N-dimensional tensors** with shape, strides and contiguous buffer.
- **Reverse-mode autodiff** through a dynamically built computation graph
- **operators:** '+', '-', '*', 'matmul', 'sum', 'mean', 'transpose'(contiguous)
- **Broadcasting** built only for matmul
- **Gradient accumulation** for variables used multiple times in the graph.
- **Manual SGD** via ```void SGD_step(double lr)``` and ```SGD_step_for_scalar(double lr, double g)```
- **Built-in tests** for all the operations both tensor and Value

## Example models
- **Linear Regression on synthetic data (multi-feature)**

## Build
Requires CMake >= 3.15 and C++-20 capable compiler
### to build:
```bash
mkdir build && cd build
cmake ..
make
```
### to run tests:
```bash
ctest
```
### to run linear regression:
```bash
./examples/linear_regression
```
