#include "autograd/tensor.hpp"
#include "autograd/ops.hpp"
#include <cmath>
#include <iostream>

int main()
{
    using namespace ag;

    auto A = std::make_shared<Tensor>(
        std::vector<double>{1,2,3,4,5,6}, std::vector<size_t>{2,3});

    std::cout << "A shape    : ";
    for (auto d : A->shape())   std::cout << d << " ";
    std::cout << "\n";

    std::cout << "A strides  : ";
    for (auto d : A->strides()) std::cout << d << " ";
    std::cout << "\n";

    std::cout << "A values sz: " << A->values().size() << "\n";

    auto B = std::make_shared<Tensor>(
        std::vector<double>{7,8,9,10,11,12}, std::vector<size_t>{3,2});

    std::cout << "B shape    : ";
    for (auto d : B->shape())   std::cout << d << " ";
    std::cout << "\n";

    std::cout << "B strides  : ";
    for (auto d : B->strides()) std::cout << d << " ";
    std::cout << "\n";

    std::cout << "B values sz: " << B->values().size() << "\n";

    std::cout << "\nCalling matmul...\n";
    auto C = matmul(A, B);

    const std::vector<size_t> expected_shape{2, 2};
    const std::vector<double> expected_values{58.0, 64.0, 139.0, 154.0};
    if (!C || C->shape() != expected_shape ||
        C->values().size() != expected_values.size()) {
        return 1;
    }
    for (size_t i = 0; i < expected_values.size(); ++i) {
        if (std::abs(C->values()[i] - expected_values[i]) > 1e-12) {
            return 1;
        }
    }

    std::cout << "C shape    : ";
    for (auto d : C->shape())   std::cout << d << " ";
    std::cout << "\n";
    std::cout << "C values   : ";
    for (auto v : C->values())  std::cout << v << " ";
    std::cout << "\n";

    return 0;
}