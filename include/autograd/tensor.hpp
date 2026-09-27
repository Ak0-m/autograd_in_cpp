#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <stdexcept>

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

    std::function<void(Tensor&)> backward_func;

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

    void add_grad(const std::vector<double> &g)
    {
        if (g.size() != grads_.size())
            throw std::runtime_error("add_grad: size mismatch");
        for (size_t i = 0; i < grads_.size(); ++i)
            grads_[i] += g[i];
    }

    void zero_grad();
    void backward();

    std::shared_ptr<Tensor> transpose(size_t d1, size_t d2) ;
    std::shared_ptr<Tensor> transpose() ;

    void GD_step(double lr);
    void GD_step_for_scalar(double lr, double g);
};

namespace detail
{
std::vector<size_t> compute_strides(const std::vector<size_t> &shape);

std::vector<size_t> broadcast(std::vector<size_t> a, std::vector<size_t> b);
std::vector<size_t> unravel(size_t flat, const std::vector<size_t>& shape);

std::vector<size_t> new_stride(std::vector<size_t> shape, std::vector<size_t> stride, std::vector<size_t> broad_shape);
std::vector<double> reduce_to_shape(std::vector<double> broad_grad, std::vector<size_t> broad_shape, std::vector<size_t> stride, std::vector<size_t> shape);
} // namespace detail

} // namespace ag