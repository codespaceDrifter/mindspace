#include "standard_optimizer.hpp"

StandardOptimizer::StandardOptimizer (Block* model, float learning_rate, float weight_decay, float grad_limit)
    : Optimizer(model, learning_rate, weight_decay, grad_limit, 0.0f) {}

void StandardOptimizer::step(){
    for (Tensor* param : params){
        if (param->requires_grad == true){
            Tensor* grad_step = param->grad->mul(this->learning_rate);
            Tensor* grad_step_max = grad_step->min(this->max_grad);
            Tensor* grad_step_min = grad_step_max->max(this->min_grad);

            Tensor* weight_decay_step = param->mul(this->weight_decay);
            Tensor* update = grad_step_min->add(weight_decay_step);
            param->minus_(update);
            delete grad_step;
            delete grad_step_max;
            delete grad_step_min;
            delete weight_decay_step;
            delete update;
        }
    }
}
