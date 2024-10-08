#include "relu_layer.hpp"

ReluLayer::ReluLayer(){
    this->type = "RELU_LAYER";
    this->register_block<ReluLayer> ();

    this->zero = Tensor::make_a_num(0);
    this->parameters.push_back (this->zero);
}


Tensor* ReluLayer::forward(Tensor* input) {
    Tensor* result = input->max(this->zero);
    return result;
}


void ReluLayer::init_members(){
    assert (this->parameters.size() == 1);
    this->zero = this->parameters[0];
}