#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"

Tensor* MSEloss (Tensor* pred, Tensor* target) {

    Tensor* diff = pred->minus(target);
    Tensor* squared = diff->pow(2);
    std::vector<int> target_shape = {1};
    Tensor* loss = squared->reduce_sum(target_shape);  
    
    return loss;
}

#endif