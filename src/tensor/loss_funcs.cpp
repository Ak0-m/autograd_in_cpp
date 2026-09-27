#include "autograd/loss_funcs.hpp"
#include "autograd/ops.hpp"

namespace ag
{

std::shared_ptr<Tensor> MSE_loss(std::shared_ptr<Tensor> targets, std::shared_ptr<Tensor> preds)
{
    auto diff = preds - targets;
    auto loss = mean(diff * diff);
    return loss;
}

} // namespace ag
