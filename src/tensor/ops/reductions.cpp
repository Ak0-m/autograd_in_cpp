#include "autograd/ops.hpp"

namespace ag
{

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

    s /= x->values().size();

    std::vector<std::shared_ptr<Tensor>> prev = {x};

    auto out = std::make_shared<Tensor>(std::vector<double>{s}, std::move(prev), "mean", std::vector<size_t>{1});

    out->backward_func = [x, out]() {
        double scale = out->grads()[0] / static_cast<double>(x->values().size());
        std::vector<double> contrib(x->values().size(), scale);
        x->add_grad(contrib);
    };

    return out;
}
} // namespace ag