#pragma once

#include "tensor.hpp"
#include <memory>


namespace ag
{
inline std::shared_ptr<Tensor> scalar_like(const std::shared_ptr<Tensor> &t, double v)
{
    return std::make_shared<Tensor>(std::vector<double>(t->values().size(), v), t->shape());
}

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor> &x);
std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor> &x);

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs);
inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &lhs, double rhs)
{
    return lhs + scalar_like(lhs, rhs);
}
inline std::shared_ptr<Tensor> operator+(double lhs, const std::shared_ptr<Tensor> &rhs)
{
    return scalar_like(rhs, lhs) + rhs;
}
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs);
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &lhs, double rhs)
{
    return lhs * scalar_like(lhs, rhs);
}
inline std::shared_ptr<Tensor> operator*(double lhs, const std::shared_ptr<Tensor> &rhs)
{
    return scalar_like(rhs, lhs) * rhs;
}

} // namespace ag