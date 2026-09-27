#include "autograd/loss_funcs.hpp"
#include "autograd/ops.hpp"
#include "autograd/tensor.hpp"

#include <iostream>
#include <random>

int main()
{
    const size_t N = 1000;
    const size_t F = 5;
    const size_t C = 3;

    const double lr = 0.01;
    std::cout << lr << "\n";
    const size_t EPOCH = 10000;

    std::vector<double> true_ws(F * C);
    std::vector<double> true_b(C);

    std::mt19937 seed(67);
    std::uniform_real_distribution<double> x_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> w_dist(-1.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.3);

    for (size_t i = 0; i < F * C; ++i)
    {
        true_ws[i] = w_dist(seed);
    }
    for (size_t c = 0; c < C; ++c)
    {
        true_b[c] = w_dist(seed);
    }

    std::vector<double> x(N * F);
    std::vector<double> y(N, 0);

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < F; ++j)
        {
            x[i * F + j] = x_dist(seed);
        }

        size_t best = 0;
        double best_val = -1e18;

        for (size_t c = 0; c < C; ++c)
        {
            double z = true_b[c] + noise(seed);

            for (size_t j = 0; j < F; ++j)
            {
                z += true_ws[j * C + c] * x[i * F + j];
            }

            if (z > best_val)
            {
                best_val = z;
                best = c;
            }
        }

        y[i] = static_cast<double>(best);
    }

    auto X = std::make_shared<ag::Tensor>(x, std::vector<size_t>{N, F});
    auto Y = std::make_shared<ag::Tensor>(y, std::vector<size_t>{N});

    std::vector<double> weights(F*C);
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = w_dist(seed);
    }
    std::vector<double> bias(C);
    for (size_t i = 0; i < bias.size(); ++i)
    {
        bias[i] = w_dist(seed);
    }


    auto W = std::make_shared<ag::Tensor>(weights, std::vector<size_t>{F, C});
    auto b = std::make_shared<ag::Tensor>(bias, std::vector<size_t>{1, C});

    std::cout << W->values()[0] << "\n";

    for (size_t e = 0; e < EPOCH; ++e)
    {
        auto pred = matmul(X, W) + b;
        auto loss = cross_entropy_loss(Y, pred);

        W->zero_grad();
        b->zero_grad();
        loss->backward();

        W->GD_step(lr);
        b->GD_step(lr);

        if (e % 200 == 0)
        {
            std::cout << "epoch " << e << "   loss=" << loss->values()[0] << "\n";
        }
    }

    auto preds = matmul(X, W) + b;
    size_t correct = 0;
    for (size_t i = 0; i < N; ++i)
    {
        size_t best = 0;
        double best_val = preds->values()[i * C];
        for (size_t c = 1; c < C; ++c)
        {
            double v = preds->values()[i * C + c];
            if (v > best_val)
            {
                best_val = v;
                best = c;
            }
        }
        if (best == static_cast<size_t>(Y->values()[i]))
            ++correct;
    }

    std::cout << "Accuracy: " << static_cast<double>(correct) / N << "\n";

    return 0;
}