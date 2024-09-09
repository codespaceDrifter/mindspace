#include "standard_optimizer.hpp"

void standardOptimizer::update(){
    std::vector<Tensor*> params = this->model->parameters();
    for (Tensor* param : params){
        TensorCore* update_val = param->grad_core->mul(this->neg_lr);
        param->value_core.reset(param->value_core->add(update_val));
        delete update_val;
    }
}