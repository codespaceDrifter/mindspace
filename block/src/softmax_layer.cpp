#include "softmax_layer.hpp"


SoftmaxLayer::SoftmaxLayer(){
    this->type = "SOFTMAX_LAYER";
    this->register_block<SoftmaxLayer> ();
}



Tensor* SoftmaxLayer::forward_(Tensor* input, Tensor* input2){
    Tensor* e = Tensor::make_a_num(std::exp(1.0f));
    e->requires_grad = false;
    e->operands.resize(1);
    Tensor* e_pow_input = e->pow(input);
    Tensor* e_pow_input_reduce = e_pow_input->reduce_sum(-1);
    Tensor* result = e_pow_input -> div (e_pow_input_reduce);
    return result;
}


void SoftmaxLayer::init_members(){
    //intentionally left blank
}