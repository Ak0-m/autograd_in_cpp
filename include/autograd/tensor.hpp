#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace ag
{

class Tensor : public std::enable_shared_from_this<Tensor>
{
    std::vector<double> values_;
    std::vector<double> grads_;
    std::vector<size_t> shape_;

    std::vector<size_t> strides_;
    bool contiguous_ = true;

    std::vector<std::shared_ptr<Tensor>> prev_;
    std::string oper_;

    static void build_topo(const std::shared_ptr<Tensor> &node, std::unordered_set<Tensor *> &visited,
                           std::vector<std::shared_ptr<Tensor>> &topo);
    static std::vector<std::shared_ptr<Tensor>> build_topo(const std::shared_ptr<Tensor> &node);

  public:
    explicit Tensor(std::vector<double> values, std::vector<size_t> shape);
    Tensor(std::vector<double> values, std::vector<std::shared_ptr<Tensor>> prev, std::string oper,
           std::vector<size_t> shape);

    const std::vector<double> &values() const
    {
        return values_;
    }
    const std::vector<double> &grads() const
    {
        return grads_;
    }
    const std::vector<size_t> &shape() const
    {
        return shape_;
    }
    const std::vector<size_t> &strides() const
    {
        return strides_;
    }
    bool contiguous() const
    {
        return contiguous_;
    }
    const std::vector<std::shared_ptr<Tensor>> &prev() const
    {
        return prev_;
    }
    std::string oper() const
    {
        return oper_;
    }

    void add_grad(const std::vector<double> &g);
    void zero_grad();
    void backward();

    std::function<void()> backward_func;
};

inline std::shared_ptr<Tensor> scalar_like(const std::shared_ptr<Tensor> &t, double v)
{
    return std::make_shared<Tensor>(std::vector<double>(t->values().size(), v), t->shape());
}

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

}