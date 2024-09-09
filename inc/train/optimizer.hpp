#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "block.hpp"

class Optimizer{
public:

Optimizer(Block* model, float lr) {
    this->model = model;
    //neg because want loss to go lower, save the memory to mul -1 each time
    this->neg_lr = TensorCore::make_a_num(-lr);
}

void step(){
    this->update();
    this->zero_grad();
}

virtual void update() = 0;

void zero_grad(){
    std::vector<Tensor*> params = this->model->parameters();
    for (Tensor* param : params){
        param->grad_core->fill(0);
    }
}

TensorCore* neg_lr;
Block* model;
};

#endif