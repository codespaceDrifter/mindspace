#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "block.hpp"

class Optimizer{
public:
    Optimizer (Block* model, float learning_rate, float weight_decay, float grad_limit, float momentum){
        this->model = model;
        this->params = model->get_all_tensors();
        this->learning_rate = Tensor::make_a_num(learning_rate);   
        this->weight_decay = Tensor::make_a_num(weight_decay);
        this->max_grad = Tensor::make_a_num(grad_limit);
        this->min_grad = Tensor::make_a_num(-grad_limit);
        this->momentum = Tensor::make_a_num(momentum);
    }

    ~Optimizer (){
        delete this->learning_rate;
        delete this->weight_decay;
        delete this->max_grad;
        delete this->momentum;
    }

    void zero_grad(){
        for (Tensor* param : params){
            if (param->requires_grad == true){
                delete param->grad;
                param->grad = nullptr;
            }
        }
    }

    virtual void step() = 0;

    Block* model;
    std::vector<Tensor*> params;
    Tensor* learning_rate; 
    Tensor* weight_decay;
    Tensor* max_grad;
    Tensor* min_grad;
    Tensor* momentum;
};

#endif