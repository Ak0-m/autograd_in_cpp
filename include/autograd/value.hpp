#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace ag
{

class Value
{
    double value_;
    double grad_ = 0;
    std::vector<std::shared_ptr<Value>> prev_;
    std::string oper_;

    static void build_topo(const std::shared_ptr<Value> &node, std::vector<std::shared_ptr<Value>> &out,
                           std::unordered_set<Value *> &visited);

  public:
    explicit Value(double value);
    Value(double value, std::vector<std::shared_ptr<Value>> prev, std::string oper);

    double value() const
    {
        return value_;
    }
    double grad() const
    {
        return grad_;
    }
    std::vector<std::shared_ptr<Value>> prev() const
    {
        return prev_;
    }
    std::string oper() const
    {
        return oper_;
    }

    void set_grad(double grad)
    {
        grad_ = grad;
    }
    void zero_grad();
    void backward();

    std::function<void()> backward_func;
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);

std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);

inline std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, double rhs)
{
    return lhs + std::make_shared<Value>(rhs);
}

inline std::shared_ptr<Value> operator+(double lhs, const std::shared_ptr<Value> &rhs)
{
    return std::make_shared<Value>(lhs) + rhs;
}

inline std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs, double rhs)
{
    return lhs * std::make_shared<Value>(rhs);
}

inline std::shared_ptr<Value> operator*(double lhs, const std::shared_ptr<Value> &rhs)
{
    return std::make_shared<Value>(lhs) * rhs;
}

}