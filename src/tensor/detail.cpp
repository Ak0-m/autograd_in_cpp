#include "autograd/tensor.hpp"

namespace ag
{
namespace detail
{

std::vector<size_t> compute_strides(const std::vector<size_t> &shape)
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

std::vector<size_t> broadcast(std::vector<size_t> a, std::vector<size_t> b)
{
    std::vector<size_t> out(std::max(a.size(), b.size()));

    size_t ai;
    size_t bi;

    for (size_t i = 0; i < out.size(); ++i)
    {
        if (i < out.size() - a.size())
        {
            ai = 1;
        }
        else
        {
            ai = a[i - (out.size() - a.size())];
        }
        if (i < out.size() - b.size())
        {
            bi = 1;
        }
        else
        {
            bi = b[i - (out.size() - b.size())];
        }

        if (ai == bi || ai == 1 || bi == 1)
        {
            out[i] = std::max(ai, bi);
        }
        else
        {
            throw std::runtime_error("matmul: batch dims not broadcastable");
        }
    }
    return out;
}

std::vector<size_t> unravel(size_t flat, const std::vector<size_t> &shape)
{
    std::vector<size_t> idx(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        idx[i] = flat % shape[i];
        flat /= shape[i];
    }
    return idx;
}

std::vector<size_t> new_stride(std::vector<size_t> shape, std::vector<size_t> stride, std::vector<size_t> broad_shape)
{
    size_t broad_rank = broad_shape.size();
    size_t og_rank = shape.size();
    size_t diff = broad_rank - og_rank;

    std::vector<size_t> nstride(broad_rank, 0);

    for (size_t i = 0; i < broad_rank; ++i)
    {
        if (i < diff)
        {
            continue;
        }
        if(broad_shape[i] == shape[i-diff])
        {
            nstride[i] = stride[i-diff];
        }
        if(shape[i-diff] == 1 && broad_shape[i] > 1)
        {
            nstride[i] = 0;
        }
    }

    return nstride;
}

std::vector<double> reduce_to_shape(std::vector<double> broad_grad, std::vector<size_t> broad_shape, std::vector<size_t> stride, std::vector<size_t> shape)
{
    size_t grad_elem = 1;
    
    for(size_t i = 0; i < shape.size(); ++i)
    {
        grad_elem *= shape[i];
    }

    std::vector<double> grad(grad_elem, 0.0);

    auto nstride = new_stride(shape, stride, broad_shape);

    size_t elem = 1;
    
    for(size_t i = 0; i < broad_shape.size(); ++i)
    {
        elem *= broad_shape[i];
    }

    for(size_t i = 0; i < elem; ++i)
    {
        auto idx = unravel(i, broad_shape);

        size_t idx_o = 0;
        
        for(size_t i = 0; i < broad_shape.size(); ++i)
        {
            idx_o += idx[i]*nstride[i];
        }

        grad[idx_o] += broad_grad[i];
    }

    return grad;
}

} // namespace detail

} // namespace ag
