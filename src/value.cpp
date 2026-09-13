#include "autograd/value.hpp"
#include <algorithm>
#include <set>

namespace ag
{
Value::Value(double value) : value_(value), grad_(0.0), prev_(), oper_("leaf")
{
}

Value::Value(double value, std::vector<std::shared_ptr<Value>> prev, std::string oper)
    : value_(value), grad_(0.0), prev_(prev), oper_(oper)
{
}

void Value::build_topo(const std::shared_ptr<Value> &node, std::unordered_set<Value *> &visited,
                       std::vector<std::shared_ptr<Value>> &topo)
{
    if (visited.contains(node.get()))
    {
        return;
    }

    visited.insert(node.get());

    for (auto prevN : node->prev())
    {
        build_topo(prevN, visited, topo);
    }

    topo.push_back(node);
}

std::vector<std::shared_ptr<Value>> Value::build_topo(const std::shared_ptr<Value> &node)
{
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<Value *> visited;
    build_topo(node, visited, topo);
    return topo;
}

void Value::backward()
{
    if (this->grad_ != 0)
    {
        throw std::runtime_error("backward() called on a node with non-zero gradient. "
                                 "Call zero_grad() first.");
    }

    this->grad_ = 1.0;

    std::vector<std::shared_ptr<Value>> topo = build_topo(shared_from_this());
    std::reverse(topo.begin(), topo.end());
    for (auto node : topo)
    {
        if (node->backward_func)
        {
            node->backward_func();
        }
    }
}

void Value::zero_grad()
{
    std::vector<std::shared_ptr<Value>> topo = build_topo(shared_from_this());
    for (auto node : topo)
    {
        node->grad_ = 0;
    }
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs)
{
    std::vector<std::shared_ptr<Value>> prev = {lhs, rhs};

    auto out = std::make_shared<Value>(lhs->value() + rhs->value(), prev, "+");

    out->backward_func = [lhs, rhs, out]() {
        lhs->set_grad(lhs->grad() + out->grad());
        rhs->set_grad(rhs->grad() + out->grad());
    };

    return out;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs)
{
    std::vector<std::shared_ptr<Value>> prev = {lhs, rhs};

    auto out = std::make_shared<Value>(lhs->value() * rhs->value(), prev, "*");

    out->backward_func = [lhs, rhs, out]() {
        lhs->set_grad(lhs->grad() + out->grad() * rhs->value());
        rhs->set_grad(rhs->grad() + out->grad() * lhs->value());
    };

    return out;
}
}