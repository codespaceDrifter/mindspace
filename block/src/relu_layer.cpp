#include "relu_layer.hpp"

ReluLayer::ReluLayer(){
    this->type = "RELU_LAYER";
    this->register_block<ReluLayer> ();
}


Tensor* ReluLayer::forward_(Tensor* input, Tensor* input2){
    Tensor* result = input->max(0.0);
    return result;
}


void ReluLayer::init_members(){
    //intentionally left blank
}