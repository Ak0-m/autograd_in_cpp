#include "autograd/value.hpp"

namespace ag
{
Value::Value(double value) : value_(value), grad_(0.0), prev_(), oper_("leaf")
{
}

Value::Value(double value, std::vector<std::shared_ptr<Value>> prev, std::string oper)
    : value_(value), grad_(0.0), prev_(prev), oper_(oper)
{
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs)
{
    std::vector<std::shared_ptr<Value>> prev = {lhs, rhs};

    auto out = std::make_shared<Value>(lhs->value() + rhs->value(), prev, "+");

    return out;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs)
{
    std::vector<std::shared_ptr<Value>> prev = {lhs, rhs};

    auto out = std::make_shared<Value>(lhs->value() * rhs->value(), prev, "*");

    return out;
}
}