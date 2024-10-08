#include "dense_layer.hpp"

DenseLayer::DenseLayer(int input, int output){
    this->type = "DENSE_LAYER";
    this->register_block<DenseLayer>();

    this->weight = new Tensor (input, output);
    this->bias = new Tensor (1,output);
    this->parameters.push_back(this->weight);
    this->parameters.push_back(this->bias);
}

Tensor* DenseLayer::forward(Tensor* input) {
    Tensor * A = input -> matmul (this->weight);
    Tensor * result = A-> add (bias);
    return result;
}


void DenseLayer::init_members(){
    assert (this->parameters.size() == 2);
    this->weight = this->parameters[0];
    this->bias = this->parameters[1];
}