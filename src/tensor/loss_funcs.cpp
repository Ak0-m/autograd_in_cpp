#include "autograd/loss_funcs.hpp"
#include "autograd/ops.hpp"
#include <cmath>

namespace ag
{

std::shared_ptr<Tensor> MSE_loss(std::shared_ptr<Tensor> targets, std::shared_ptr<Tensor> preds)
{
    auto diff = preds - targets;
    auto loss = mean(diff * diff);
    return loss;
}

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> targets, std::shared_ptr<Tensor> preds)
{
    if (preds->shape().size() != 2)
    {
        throw std::runtime_error("cross_entropy: preds must be 2D {N, C}");
    }
    if (targets->shape().size() != 1 || targets->shape()[0] != preds->shape()[0])
    {
        throw std::runtime_error("cross_entropy: targets must be {N}");
    }

    std::vector<double> probs(preds->values().size());
    double loss = 0.0;

    for (size_t i = 0; i < preds->shape()[0]; i++)
    {
        double m = preds->values()[i * preds->strides()[0]];

        for (size_t j = 0; j < preds->shape()[1]; j++)
        {
            m = std::max(m, preds->values()[i * preds->strides()[0] + j * preds->strides()[1]]);
        }

        double s = 0.0;

        for (size_t j = 0; j < preds->shape()[1]; j++)
        {
            probs[i * preds->strides()[0] + j * preds->strides()[1]] =
                std::exp(preds->values()[i * preds->strides()[0] + j * preds->strides()[1]] - m);
            s += probs[i * preds->strides()[0] + j * preds->strides()[1]];
        }

        for (size_t j = 0; j < preds->shape()[1]; j++)
        {
            probs[i * preds->strides()[0] + j * preds->strides()[1]] =
                probs[i * preds->strides()[0] + j * preds->strides()[1]] / s;
        }

        size_t label = static_cast<size_t>(targets->values()[i]);

        loss += -std::log(probs[i * preds->strides()[0] + label * preds->strides()[1]]);
    }

    loss /= preds->shape()[0];

    auto out = std::make_shared<Tensor>(std::vector<double>{loss}, std::vector<std::shared_ptr<Tensor>>{preds},
                                        "cross entropy", std::vector<size_t>{1});

    out->backward_func = [preds, probs, targets](Tensor &out) {
    std::vector<double> grads(preds->values().size());
    // Capture the incoming gradient from the loss tensor
    double next_grad = out.grads()[0]; 

    for (size_t i = 0; i < preds->shape()[0]; ++i) {
        size_t label = static_cast<size_t>(targets->values()[i]);
        for (size_t j = 0; j < preds->shape()[1]; ++j) {
            double onehot = (j == label) ? 1.0 : 0.0;
            
            // Multiply by next_grad to maintain the chain rule
            grads[i * preds->strides()[0] + j * preds->strides()[1]] = 
                ((probs[i * preds->strides()[0] + j * preds->strides()[1]] - onehot) / preds->shape()[0]) * next_grad;
        }
    }
    preds->add_grad(grads);
};

    return out;
}

} // namespace ag
