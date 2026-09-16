#include "autograd/ops.hpp"

namespace ag
{

static std::vector<size_t> broadcast(std::vector<size_t> a, std::vector<size_t> b)
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

static std::vector<size_t> unravel(size_t flat, const std::vector<size_t>& shape)
{
    std::vector<size_t> idx(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        idx[i] = flat % shape[i];
        flat  /= shape[i];
    }
    return idx;
}

static size_t compute_offset(const std::vector<size_t> &shape, const std::vector<size_t> &strides,
                             const std::vector<size_t> &batchi)
{

    size_t align = batchi.size() - shape.size() - 2;
    size_t offset = 0;

    for (size_t i = 0; i < shape.size() - 2; ++i)
    {
        size_t outD = align + i;
        size_t idx;
        if (shape[i] == 1)
        {
            idx = 0;
        }
        else
        {
            idx = batchi[outD];
        }
        offset += idx * strides[i];
    }
    return offset;
}

std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs)
{
    if (lhs->shape().size() < 2 || rhs->shape().size() < 2)
    {
        throw std::runtime_error("matmul: tensors must be at least 2D");
    }

    const auto &as = lhs->shape();
    const auto &bs = rhs->shape();

    size_t m = as[as.size() - 2];
    size_t k1 = as[as.size() - 1];
    size_t k2 = bs[bs.size() - 2];
    size_t n = bs[bs.size() - 1];

    if (k1 != k2)
    {
        throw std::runtime_error("matmul: inner dimension mismatch");
    }

    std::vector<size_t> a_batch(as.begin(), as.end() - 2);
    std::vector<size_t> b_batch(bs.begin(), bs.end() - 2);
    std::vector<size_t> out_batch = broadcast(a_batch, b_batch);

    std::vector<size_t> out_shape = out_batch;
    out_shape.push_back(m);
    out_shape.push_back(n);

    size_t out_size = 1;
    for (auto i : out_shape)
    {
        out_size *= i;
    }

    std::vector<size_t> out_strides(out_shape.size());
    {
        size_t s = 1;
        for (size_t i = out_shape.size(); i-- > 0;)
        {
            out_strides[i] = s;
            s *= out_shape[i];
        }
    }

    std::vector<double> out_values(out_size, 0.0);

    size_t total_batch = 1;
    for (auto d : out_batch)
    {
        total_batch *= d;
    }

    for (size_t d = 0; d < total_batch; ++d)
    {
        std::vector<size_t> out_idx = unravel(d, out_batch);

        size_t a_offset = compute_offset(lhs->shape(), lhs->strides(), out_idx);
        size_t b_offset = compute_offset(rhs->shape(), rhs->strides(), out_idx);
        size_t c_offset = compute_offset(out_shape, out_strides, out_idx);

        size_t a_rs = lhs->strides()[as.size() - 2];
        size_t a_cs = lhs->strides()[as.size() - 1];
        size_t b_rs = rhs->strides()[bs.size() - 2];
        size_t b_cs = rhs->strides()[bs.size() - 1];
        size_t c_rs = out_strides[out_shape.size() - 2];
        size_t c_cs = out_strides[out_shape.size() - 1];

        for (size_t i = 0; i < m; ++i)
        {
            for (size_t k = 0; k < k1; ++k)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    double a_val = lhs->values()[a_offset + i * a_rs + k * a_cs];
                    double b_val = rhs->values()[b_offset + k * b_rs + j * b_cs];
                    out_values[c_offset + i * c_rs + j * c_cs] += a_val * b_val;
                }
            }
        }
    }

    std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};

    auto out = std::make_shared<Tensor>(std::move(out_values), std::move(prev), "@", out_shape);

    out->backward_func = [lhs, rhs, out, m, n, k1, out_batch, total_batch, out_shape, out_strides, as, bs]() {
        std::vector<double> lhs_contrib(lhs->values().size(), 0.0);
        std::vector<double> rhs_contrib(rhs->values().size(), 0.0);

        for (size_t d = 0; d < total_batch; ++d)
        {
            std::vector<size_t> out_idx = unravel(d, out_batch);

            size_t a_offset = compute_offset(as, lhs->strides(), out_idx);
            size_t b_offset = compute_offset(bs, rhs->strides(), out_idx);
            size_t c_offset = compute_offset(out_shape, out_strides, out_idx);

            size_t a_rs = lhs->strides()[as.size() - 2];
            size_t a_cs = lhs->strides()[as.size() - 1];
            size_t b_rs = rhs->strides()[bs.size() - 2];
            size_t b_cs = rhs->strides()[bs.size() - 1];
            size_t c_rs = out_strides[out_shape.size() - 2];
            size_t c_cs = out_strides[out_shape.size() - 1];

            for (size_t i = 0; i < m; ++i)
            {
                for (size_t k = 0; k < k1; ++k)
                {
                    double acc = 0.0;
                    for (size_t j = 0; j < n; ++j)
                    {
                        double g = out->grads()[c_offset + i * c_rs + j * c_cs];
                        double b_val = rhs->values()[b_offset + k * b_rs + j * b_cs];
                        acc += g * b_val;
                    }
                    lhs_contrib[a_offset + i * a_rs + k * a_cs] += acc;
                }
            }

            for (size_t k = 0; k < k1; ++k)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    double acc = 0.0;
                    for (size_t i = 0; i < m; ++i)
                    {
                        double av = lhs->values()[a_offset + i * a_rs + k * a_cs];
                        double g = out->grads()[c_offset + i * c_rs + j * c_cs];
                        acc += av * g;
                    }
                    rhs_contrib[b_offset + k * b_rs + j * b_cs] += acc;
                }
            }
        }

        lhs->add_grad(lhs_contrib);
        rhs->add_grad(rhs_contrib);
    };

    return out;
}
} // namespace ag