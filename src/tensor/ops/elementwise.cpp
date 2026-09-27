#include "autograd/ops.hpp"
#include <stdexcept>

namespace ag
{
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs)
{
    if (lhs->shape() == rhs->shape())
    {
        std::vector<double> out_values(lhs->values().size());

        for (size_t i = 0; i < out_values.size(); ++i)
        {
            out_values[i] = lhs->values()[i] + rhs->values()[i];
        }

        std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};

        auto out = std::make_shared<Tensor>(std::move(out_values), std::move(prev), "+", lhs->shape());

        out->backward_func = [lhs, rhs](Tensor &out) {
            lhs->add_grad(out.grads());
            rhs->add_grad(out.grads());
        };
        return out;

    /**/ }
    else
    {
        auto broad_shape = detail::broadcast(lhs->shape(), rhs->shape());
        auto lhs_stride = detail::new_stride(lhs->shape(), lhs->strides(), broad_shape);
        auto rhs_stride = detail::new_stride(rhs->shape(), rhs->strides(), broad_shape);

        size_t total = 1;

        for (size_t i = 0; i < broad_shape.size(); ++i)
        {
            total *= broad_shape[i];
        }

        std::vector<double> broad_values(total);

        for (size_t i = 0; i < total; ++i)
        {
            auto idx = detail::unravel(i, broad_shape);

            size_t a_off = 0, b_off = 0;

            for (size_t j = 0; j < broad_shape.size(); ++j)
            {
                a_off += idx[j] * lhs_stride[j];
                b_off += idx[j] * rhs_stride[j];
            }

            broad_values[i] = lhs->values()[a_off] + rhs->values()[b_off];
        }

        std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};
        auto out = std::make_shared<Tensor>(std::move(broad_values), prev, "+", broad_shape);

        out->backward_func = [lhs, rhs](Tensor &out) {
            auto lg = detail::reduce_to_shape(out.grads(), out.shape(), lhs->strides(), lhs->shape());
            auto rg = detail::reduce_to_shape(out.grads(), out.shape(), rhs->strides(), rhs->shape());
            lhs->add_grad(lg);
            rhs->add_grad(rg);
        };
        return out;
    }
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs)
{
    if (lhs->shape() == rhs->shape())
    {
        std::vector<double> out_values(lhs->values().size());

        for (size_t i = 0; i < out_values.size(); ++i)
        {
            out_values[i] = lhs->values()[i] - rhs->values()[i];
        }

        std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};

        auto out = std::make_shared<Tensor>(std::move(out_values), std::move(prev), "-", lhs->shape());

        out->backward_func = [lhs, rhs](Tensor &out) {
            lhs->add_grad(out.grads());

            std::vector<double> neg(out.grads().size());
            for (size_t i = 0; i < neg.size(); ++i)
            {
                neg[i] = -out.grads()[i];
            }
            rhs->add_grad(neg);
        };

        return out;
    }
    else
    {
        auto broad_shape = detail::broadcast(lhs->shape(), rhs->shape());
        auto lhs_stride = detail::new_stride(lhs->shape(), lhs->strides(), broad_shape);
        auto rhs_stride = detail::new_stride(rhs->shape(), rhs->strides(), broad_shape);

        size_t total = 1;

        for (size_t i = 0; i < broad_shape.size(); ++i)
        {
            total *= broad_shape[i];
        }

        std::vector<double> broad_values(total);

        for (size_t i = 0; i < total; ++i)
        {
            auto idx = detail::unravel(i, broad_shape);

            size_t a_off = 0, b_off = 0;

            for (size_t j = 0; j < broad_shape.size(); ++j)
            {
                a_off += idx[j] * lhs_stride[j];
                b_off += idx[j] * rhs_stride[j];
            }

            broad_values[i] = lhs->values()[a_off] - rhs->values()[b_off];
        }

        std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};
        auto out = std::make_shared<Tensor>(std::move(broad_values), prev, "-", broad_shape);

        out->backward_func = [lhs, rhs](Tensor &out) {
            auto lg = detail::reduce_to_shape(out.grads(), out.shape(), lhs->strides(), lhs->shape());
            auto rg = detail::reduce_to_shape(out.grads(), out.shape(), rhs->strides(), rhs->shape());

            for (size_t i = 0; i < rg.size(); ++i)
            {
                rg[i] = -rg[i];
            }
            lhs->add_grad(lg);
            rhs->add_grad(rg);
        };
        return out;
    }
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &lhs, const std::shared_ptr<Tensor> &rhs)
{
    if (lhs->shape() == rhs->shape())
    {
        std::vector<double> out_values(lhs->values().size());

        for (size_t i = 0; i < out_values.size(); ++i)
        {
            out_values[i] = lhs->values()[i] * rhs->values()[i];
        }

        std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};

        auto out = std::make_shared<Tensor>(std::move(out_values), std::move(prev), "*", lhs->shape());

        out->backward_func = [lhs, rhs](Tensor &out) {
            std::vector<double> contrib(out.values().size());
            for (size_t i = 0; i < contrib.size(); ++i)
            {
                contrib[i] = out.grads()[i] * rhs->values()[i];
            }
            lhs->add_grad(contrib);
            for (size_t i = 0; i < contrib.size(); ++i)
            {
                contrib[i] = out.grads()[i] * lhs->values()[i];
            }
            rhs->add_grad(contrib);
        };

        return out;
    }
    else
    {
        auto broad_shape = detail::broadcast(lhs->shape(), rhs->shape());
        auto lhs_stride = detail::new_stride(lhs->shape(), lhs->strides(), broad_shape);
        auto rhs_stride = detail::new_stride(rhs->shape(), rhs->strides(), broad_shape);

        size_t total = 1;

        for (size_t i = 0; i < broad_shape.size(); ++i)
        {
            total *= broad_shape[i];
        }

        std::vector<double> broad_values(total);

        for (size_t i = 0; i < total; ++i)
        {
            auto idx = detail::unravel(i, broad_shape);

            size_t a_off = 0, b_off = 0;

            for (size_t j = 0; j < broad_shape.size(); ++j)
            {
                a_off += idx[j] * lhs_stride[j];
                b_off += idx[j] * rhs_stride[j];
            }

            broad_values[i] = lhs->values()[a_off] * rhs->values()[b_off];
        }

        std::vector<std::shared_ptr<Tensor>> prev = {lhs, rhs};
        auto out = std::make_shared<Tensor>(std::move(broad_values), prev, "*", broad_shape);

        out->backward_func = [lhs, rhs, broad_shape, lhs_stride, rhs_stride](Tensor &out) {
            size_t total = 1;
            for (size_t i = 0; i < broad_shape.size(); ++i)
            {
                total *= broad_shape[i];
            }

            std::vector<double> tmp(total);

            for (size_t i = 0; i < total; ++i)
            {
                auto idx = detail::unravel(i, broad_shape);

                size_t b_off = 0;

                for (size_t j = 0; j < broad_shape.size(); ++j)
                {
                    b_off += idx[j] * rhs_stride[j];
                }

                tmp[i] = out.grads()[i] * rhs->values()[b_off];
            }
            auto lg = detail::reduce_to_shape(tmp, out.shape(), lhs->strides(), lhs->shape());

            for (size_t i = 0; i < total; ++i)
            {
                auto idx = detail::unravel(i, broad_shape);

                size_t a_off = 0;

                for (size_t j = 0; j < broad_shape.size(); ++j)
                {
                    a_off += idx[j] * lhs_stride[j];
                }

                tmp[i] = out.grads()[i] * lhs->values()[a_off];
            }
            auto rg = detail::reduce_to_shape(tmp, out.shape(), rhs->strides(), rhs->shape());

            lhs->add_grad(lg);
            rhs->add_grad(rg);
        };
        return out;
    }
}
} // namespace ag
