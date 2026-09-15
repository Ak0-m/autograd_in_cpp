#include "autograd/tensor.hpp"
#include "autograd/ops.hpp"
#include <algorithm>
#include <set>

namespace ag
{

Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape)
    : values_(std::move(values)), grads_(values_.size(), 0.0), shape_(std::move(shape)),
      strides_(detail::compute_strides(shape_)), oper_("leaf")
{
}

Tensor::Tensor(std::vector<double> values, std::vector<std::shared_ptr<Tensor>> prev, std::string oper,
               std::vector<size_t> shape)
    : values_(std::move(values)), grads_(values_.size(), 0.0), shape_(std::move(shape)),
      strides_(detail::compute_strides(shape_)), prev_(std::move(prev)), oper_(std::move(oper))
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

} // namespace ag