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

    std::function<void()> backward_func;

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

    void SGD_step(double lr);
    void SGD_step_for_scalar(double lr, double g);
};

namespace detail
{
inline std::vector<size_t> compute_strides(const std::vector<size_t> &shape)
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
} // namespace detail

} // namespace ag