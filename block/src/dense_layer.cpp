#include "dense_layer.hpp"

DenseLayer::DenseLayer(int input, int output){
    this->type = "DENSE_LAYER";
    this->register_block<DenseLayer>();
    this->parameters.push_back(new Tensor (input, output));
    this->parameters.push_back(new Tensor (1,output));
    this->init_members();
}

Tensor* DenseLayer::forward_(Tensor* input, Tensor* input2) {
    Tensor * A = input -> matmul (this->weight);
    Tensor * result = A-> add (bias);
    return result;
}


void DenseLayer::init_members(){
    assert (this->parameters.size() == 2);
    this->weight = this->parameters[0];
    this->bias = this->parameters[1];
}