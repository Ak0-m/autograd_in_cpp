#include "autograd/ops.hpp"
#include "autograd/tensor.hpp"

#include <iostream>
#include <random>

int main()
{
    const size_t N = 1000;
    const size_t F = 3;

    const double lr = 0.001;
    const size_t EPOCH = 50000;

    std::vector<double> true_ws{10.0, 6.0, -4.0};
    const double true_b = 2.0;

    std::mt19937 seed(67);
    std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.1);

    std::vector<double> x(N * F);
    std::vector<double> y(N, 0);

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < F; ++j)
        {
            x[i * F + j] = x_dist(seed);
            y[i] += true_ws[j] * x[i * F + j];
        }
        y[i] += true_b + noise(seed);
    }

    auto X = std::make_shared<ag::Tensor>(x, std::vector<size_t>{N, F});
    auto Y = std::make_shared<ag::Tensor>(y, std::vector<size_t>{N, 1});

    auto W = std::make_shared<ag::Tensor>(std::vector<double>(F, 0.0), std::vector<size_t>{F, 1});
    auto b = std::make_shared<ag::Tensor>(std::vector<double>(N, 0.0), std::vector<size_t>{N, 1});

    double l1 = -1;

    for (size_t e = 0; e < EPOCH; ++e)
    {
        auto pred = matmul(X, W) + b;
        auto diff = pred - Y;
        auto loss = mean(diff * diff);

        if(l1 == loss->values()[0])
        {
            break;
        }
        else
        {
            l1 = loss->values()[0];
        }

        W->zero_grad();
        b->zero_grad();
        loss->backward();

        W->SGD_step(lr);

        double g = 0.0;
        for (size_t i = 0; i < N; ++i)
        {
            g += b->grads()[i];
        }

        b->SGD_step_for_scalar(lr, g);

        if (e % 200 == 0)
        {
            std::cout << "epoch " << e << "   loss=" << loss->values()[0] << "\n";
        }
    }

    std::cout << "\nLearned:\n";
    for (size_t j = 0; j < F; ++j)
        std::cout << "  w[" << j << "] = " << W->values()[j] << "   (true " << true_ws[j] << ")\n";
    std::cout << "  b    = " << b->values()[0] << "   (true " << true_b << ")\n";
}