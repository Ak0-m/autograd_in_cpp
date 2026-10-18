#include "autograd/loss_funcs.hpp"
#include "autograd/ops.hpp"
#include "autograd/tensor.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

/*
This is a MLP meant to be trained and tested on MNIST
download the mnist_test.csv and mnist.train.csv from
https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
and put them in autograd_in_cpp directory
*/

void stream(const std::string &fl, std::vector<double> &data, std::vector<double> &label)
{
    std::ifstream file(fl);
    std::string line;

    size_t i = 0;
    while (std::getline(file, line))
    {
        std::string s;
        std::stringstream ss(line);

        size_t j = 0;
        while (std::getline(ss, s, ','))
        {
            if (j == 0)
            {
                label[i] = std::stod(s);
            }
            else
            {
                data[i * 784 + j - 1] = ((std::stod(s) / 255) - 0.1307) / 0.3081;
            }
            ++j;
        }
        ++i;
    }
}

int main()
{
    const size_t IN_LAYER = 784, HL1 = 128, HL2 = 64, OUT = 10;
    std::vector<double> mnist_train_data(47040000);
    std::vector<double> mnist_train_labels(60000);
    std::vector<double> mnist_test_data(7840000);
    std::vector<double> mnist_test_labels(10000);

    stream("./mnist_train.csv", mnist_train_data, mnist_train_labels);
    stream("./mnist_test.csv", mnist_test_data, mnist_test_labels);

    std::mt19937_64 seed(67);
    std::normal_distribution<double> init1(0.0, std::sqrt(2.0 / 784));
    std::normal_distribution<double> init2(0.0, std::sqrt(2.0 / 128));
    std::normal_distribution<double> init3(0.0, std::sqrt(2.0 / 64));
    auto rand = [&seed](size_t n, auto &dist) {
        std::vector<double> v(n);
        for (size_t i = 0; i < n; ++i)
        {
            v[i] = dist(seed);
        }
        return v;
    };

    auto W1 = std::make_shared<ag::Tensor>(rand(IN_LAYER * HL1, init1), std::vector<size_t>{IN_LAYER, HL1});
    auto B1 = std::make_shared<ag::Tensor>(rand(HL1, init1), std::vector<size_t>{1, HL1});
    auto W2 = std::make_shared<ag::Tensor>(rand(HL1 * HL2, init2), std::vector<size_t>{HL1, HL2});
    auto B2 = std::make_shared<ag::Tensor>(rand(HL2, init2), std::vector<size_t>{1, HL2});
    auto W3 = std::make_shared<ag::Tensor>(rand(HL2 * OUT, init3), std::vector<size_t>{HL2, OUT});
    auto B3 = std::make_shared<ag::Tensor>(rand(OUT, init3), std::vector<size_t>{1, OUT});

    const size_t N = 60000, BATCH = 64, EPOCH = 10;
    const double lr = 0.02;

    std::vector<size_t> indices(N);

    for (size_t i = 0; i < N; ++i)
    {
        indices[i] = i;
    }

    for (size_t e = 0; e < EPOCH; ++e)
    {
        std::shuffle(indices.begin(), indices.end(), seed);

        for (size_t start = 0; start < N; start += BATCH)
        {
            std::vector<double> xb(BATCH * 784), yb(BATCH);
            size_t end = std::min(start + BATCH, N);
            size_t b = end - start;

            for (size_t i = 0; i < b; ++i)
            {
                size_t idx = indices[i + start];
                std::copy(mnist_train_data.begin() + idx * 784, mnist_train_data.begin() + idx * 784 + 784,
                          xb.begin() + i * 784);
                yb[i] = mnist_train_labels[idx];
            }

            auto X = std::make_shared<ag::Tensor>(std::move(xb), std::vector<size_t>{b, 784});
            auto Y = std::make_shared<ag::Tensor>(std::move(yb), std::vector<size_t>{b});

            auto h1 = ag::relu(ag::matmul(X, W1) + B1);
            auto h2 = ag::relu(ag::matmul(h1, W2) + B2);
            auto logits = ag::matmul(h2, W3) + B3;
            auto loss = ag::cross_entropy_loss(Y, logits);

            W1->zero_grad();
            B1->zero_grad();
            W2->zero_grad();
            B2->zero_grad();
            W3->zero_grad();
            B3->zero_grad();
            loss->backward();

            W1->GD_step(lr);
            B1->GD_step(lr);
            W2->GD_step(lr);
            B2->GD_step(lr);
            W3->GD_step(lr);
            B3->GD_step(lr);
        }
        std::cout << "epoch " << e << " done\n";
    }

    auto Xtest = std::make_shared<ag::Tensor>(std::move(mnist_test_data), std::vector<size_t>{10000, 784});

    auto h1 = ag::relu(ag::matmul(Xtest, W1) + B1);
    auto h2 = ag::relu(ag::matmul(h1, W2) + B2);
    auto logits = ag::matmul(h2, W3) + B3;

    size_t correct = 0;

    for (size_t i = 0; i < 10000; ++i)
    {
        size_t best = 0;
        double best_val = logits->values()[i * 10];

        for (size_t j = 1; j < 10; ++j)
        {
            if (logits->values()[i * 10 + j] > best_val)
            {
                best_val = logits->values()[i * 10 + j];
                best = j;
            }
        }

        if (best == mnist_test_labels[i])
        {
            ++correct;
        }
    }

    std::cout << "Test accuracy: " << (double)correct / 10000.0 << "\n";

    return 0;
}
