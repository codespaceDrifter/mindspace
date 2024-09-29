#include "relu_layer.hpp"

ReluLayer::ReluLayer(){
    this->type = "RELU_LAYER";
    this->register_block();

    this->zero = Tensor::make_a_num(0);
    this->parameters.push_back (this->zero);
}


Tensor* ReluLayer::forward(Tensor* input) {
    Tensor* result = input->max(this->zero);
    return result;
}

Block* ReluLayer::factory_create (){
    return new ReluLayer();
}


void ReluLayer::init_members(){
    assert (this->parameters.size() == 1);
    this->zero = this->parameters[0];
}