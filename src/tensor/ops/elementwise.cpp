#include "autograd/ops.hpp"
#include <stdexcept>

namespace ag
{
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs)
{
    if (lhs->shape() != rhs->shape())
    {
        throw std::runtime_error("operator+: shape mismatch");
    }

    std::vector<double> out_values(lhs->values().size());

    for (size_t i = 0; i < out_values.size(); ++i)
    {
        out_values[i] = lhs->values()[i] + rhs->values()[i];
    }

    std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};

    auto out = std::make_shared<Tensor>(std::move(out_values), std::move(prev), "+", lhs->shape());

    out->backward_func = [lhs, rhs, out]() {
        lhs->add_grad(out->grads());
        rhs->add_grad(out->grads());
    };

    return out;
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs)
{
    if (lhs->shape() != rhs->shape())
    {
        throw std::runtime_error("operator-: shape mismatch");
    }

    std::vector<double> out_values(lhs->values().size());

    for (size_t i = 0; i < out_values.size(); ++i)
    {
        out_values[i] = lhs->values()[i] - rhs->values()[i];
    }

    std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};

    auto out = std::make_shared<Tensor>(std::move(out_values), std::move(prev), "-", lhs->shape());

    out->backward_func = [lhs, rhs, out]() {
        lhs->add_grad(out->grads());
        
        std::vector<double> neg(out->grads().size());
        for(size_t i = 0; i < neg.size(); ++i)
        {
            neg[i] = -out->grads()[i];
        }
        rhs->add_grad(neg);
    };

    return out;
}


std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs)
{
    if (lhs->shape() != rhs->shape())
    {
        throw std::runtime_error("operator*: shape mismatch");
    }

    std::vector<double> out_values(lhs->values().size());

    for (size_t i = 0; i < out_values.size(); ++i)
    {
        out_values[i] = lhs->values()[i] * rhs->values()[i];
    }

    std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};

    auto out = std::make_shared<Tensor>(std::move(out_values), std::move(prev), "*", lhs->shape());

    out->backward_func = [lhs, rhs, out]() {
        std::vector<double> contrib(out->values().size());
        for (size_t i = 0; i < contrib.size(); ++i)
        {
            contrib[i] = out->grads()[i] * rhs->values()[i];
        }
        lhs->add_grad(contrib);
        for (size_t i = 0; i < contrib.size(); ++i)
        {
            contrib[i] = out->grads()[i] * lhs->values()[i];
        }
        rhs->add_grad(contrib);
    };

    return out;
}
} // namespace ag
