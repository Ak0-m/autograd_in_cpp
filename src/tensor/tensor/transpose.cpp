#include "autograd/tensor.hpp"

namespace ag
{
std::shared_ptr<Tensor> Tensor::transpose(size_t d1, size_t d2)
{
    const auto &s = shape_;
    if (d1 >= s.size() || d2 >= s.size() || d1 == d2)
    {
        throw std::runtime_error("Tensor::transpose: bad axes");
    }

    std::vector<size_t> nshape = s;
    std::swap(nshape[d1], nshape[d2]);

    std::vector<double> out_values(values_.size());

    size_t rank = nshape.size();
    std::vector<size_t> idx(rank, 0);

    for (size_t i = 0; i < out_values.size(); ++i)
    {
        size_t j = 0;

        for (size_t k = 0; k < rank; ++k)
        {
            size_t src_k = k;
            if (k == d1)
            {
                src_k = d2;
            }
            else if (k == d2)
            {
                src_k = d1;
            }
            j += idx[k] * strides_[src_k];
        }
        out_values[i] = values_[j];

        for (size_t k = rank - 1; true; --k)
        {
            if (++idx[k] < nshape[k])
                break;
            idx[k] = 0;
            if (k == 0)
                break;
        }
    }

    auto self = std::const_pointer_cast<Tensor>(shared_from_this());

    auto out = std::make_shared<Tensor>(std::move(out_values), std::vector<std::shared_ptr<Tensor>>{self}, "transpose",
                                        nshape);

    out->backward_func = [self, d1, d2, out]() {
        auto grad = std::make_shared<Tensor>(out->grads(), out->shape());
        self->add_grad(grad->transpose(d1, d2)->values());
    };

    return out;
}

std::shared_ptr<Tensor> Tensor::transpose()
{
    if (shape_.size() < 2)
    {
        return std::const_pointer_cast<Tensor>(shared_from_this());
    }

    return transpose(shape_.size() - 2, shape_.size() - 1);
}
} // namespace ag
