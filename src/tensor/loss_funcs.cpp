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

    const size_t C = preds->shape()[1];
    const size_t N = preds->shape()[0];
    const size_t r_stride = preds->strides()[0];
    const size_t c_stride = preds->strides()[1];

    const double *preds_value = preds->values().data();
    const double *targets_value = targets->values().data();
    
    std::vector<double> probs(preds->values().size());
    double loss = 0.0;

    for (size_t i = 0; i < N; i++)
    {
        double m = preds_value[i * r_stride];

        for (size_t j = 0; j < C; j++)
        {
            m = std::max(m, preds_value[i * r_stride + j * c_stride]);
        }

        double s = 0.0;

        for (size_t j = 0; j < C; j++)
        {
            probs[i * r_stride + j * c_stride] =
                std::exp(preds_value[i * r_stride + j * c_stride] - m);
            s += probs[i * r_stride + j * c_stride];
        }

        for (size_t j = 0; j < C; j++)
        {
            probs[i * r_stride + j * c_stride] =
                probs[i * r_stride + j * c_stride] / s;
        }

        size_t label = static_cast<size_t>(targets->values()[i]);

        loss += -std::log(probs[i * r_stride + label * c_stride]);
    }

    loss /= N;

    auto out = std::make_shared<Tensor>(std::vector<double>{loss}, std::vector<std::shared_ptr<Tensor>>{preds},
                                        "cross entropy", std::vector<size_t>{1});

    out->backward_func = [preds, probs = std::move(probs), targets, N, C, r_stride, c_stride](Tensor &out) {
    std::vector<double> grads(preds->values().size());
    // Capture the incoming gradient from the loss tensor
    double next_grad = out.grads()[0]; 

    for (size_t i = 0; i < N; ++i) {
        size_t label = static_cast<size_t>(targets->values()[i]);
        for (size_t j = 0; j < C; ++j) {
            double onehot = (j == label) ? 1.0 : 0.0;
            
            // Multiply by next_grad to maintain the chain rule
            grads[i * r_stride + j * c_stride] = 
                ((probs[i * r_stride + j * c_stride] - onehot) / N) * next_grad;
        }
    }
    preds->add_grad(grads);
};

    return out;
}

} // namespace ag
