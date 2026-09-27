# Autograd in C++ - A Minimal Autograd Engine

A from scratch automatic differentiation engine in c++
Built to understand how production frameworks like PyTorch work

## Why I made this
I made it as a way to learn the basics of Machine learning
that are abstracted away by frameworks like PyTorch

## What it has
- **N-dimensional tensors** with shape, strides and a flat contiguous buffer.
- **Reverse-mode autodiff** through a dynamically built computation graph
- **operators:** '+', '-', '*', 'matmul', 'sum', 'mean', 'transpose'(contiguous)
- **Broadcasting** for matmul and elementwise operations
- **Gradient accumulation** for variables used multiple times in the graph.
- **Gradient Descent** via ```Tensor::GD_step(lr)``` and ```Tensor::GD_step_for_scalar(lr, g)```
- **Built-in tests** for every operations both for bith Tensor and Value clases

## Example models
- **Linear Regression**
- **Logistic Regression**

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
