#include "autograd/tensor.hpp"
#include <algorithm>
#include <set>

namespace ag
{

static std::vector<size_t> compute_strides(const std::vector<size_t> &shape)
{
    std::vector<size_t> strides(shape.size());
    size_t s = 1;

    if (shape.empty())
        return {};

    for (size_t i = shape.size() - 1; true; --i)
    {
        strides[i] = s;
        s *= shape[i];

        if (i == 0)
        {
            break;
        }
    }
    return strides;
}

Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape)
    : values_(std::move(values)), grads_(values_.size(), 0.0), shape_(std::move(shape)),
      strides_(compute_strides(shape)), oper_("leaf")
{
}

Tensor::Tensor(std::vector<double> values, std::vector<std::shared_ptr<Tensor>> prev, std::string oper,
               std::vector<size_t> shape)
    : values_(std::move(values)), grads_(values_.size(), 0.0), shape_(std::move(shape)),
      strides_(compute_strides(shape_)), prev_(std::move(prev)), oper_(std::move(oper))
{
}

void Tensor::build_topo(const std::shared_ptr<Tensor> &node, std::unordered_set<Tensor *> &visited,
                        std::vector<std::shared_ptr<Tensor>> &topo)
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

std::vector<std::shared_ptr<Tensor>> Tensor::build_topo(const std::shared_ptr<Tensor> &node)
{
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor *> visited;
    build_topo(node, visited, topo);
    return topo;
}

void Tensor::backward()
{
    if (!(std::all_of(grads_.begin(), grads_.end(), [](double i) { return i == 0; })))
    {
        throw std::runtime_error("backward() called on a node with non-zero gradient. "
                                 "Call zero_grad() first.");
    }

    std::fill(grads_.begin(), grads_.end(), 1.0);

    std::vector<std::shared_ptr<Tensor>> topo = build_topo(shared_from_this());
    std::reverse(topo.begin(), topo.end());
    for (auto node : topo)
    {
        if (node->backward_func)
        {
            node->backward_func();
        }
    }
}

void Tensor::zero_grad()
{
    std::vector<std::shared_ptr<Tensor>> topo = build_topo(shared_from_this());
    for (auto node : topo)
    {
        std::fill(node->grads_.begin(), node->grads_.end(), 0.0);
    }
}

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor> &x)
{
    double s = 0;

    for (size_t i = 0; i < x->values().size(); i++)
    {
        s += x->values()[i];
    }

    std::vector<std::shared_ptr<Tensor>> prev = {x};

    auto out = std::make_shared<Tensor>(std::vector<double>{s}, std::move(prev), "sum", std::vector<size_t>{1});

    out->backward_func = [x, out]() { 
        std::vector<double> contrib(x->values().size(), out->grads()[0]);
        x->add_grad(contrib);
    };

    return out;
}

std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor> &x)
{
    double s = 0;

    for (size_t i = 0; i < x->values().size(); i++)
    {
        s += x->values()[i];
    }

    s/=x->values().size();

    std::vector<std::shared_ptr<Tensor>> prev = {x};

    auto out = std::make_shared<Tensor>(std::vector<double>{s}, std::move(prev), "mean", std::vector<size_t>{1});

    out->backward_func = [x, out]() { 
        double scale = out->grads()[0] / static_cast<double>(x->values().size());
        std::vector<double> contrib(x->values().size(), scale);
        x->add_grad(contrib);
    };

    return out;
}

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