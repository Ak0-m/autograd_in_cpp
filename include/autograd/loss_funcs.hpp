#pragma once
#include "tensor.hpp"

namespace ag
{
    std::shared_ptr<Tensor> MSE_loss(std::shared_ptr<Tensor> targets, std::shared_ptr<Tensor> preds);
} 
