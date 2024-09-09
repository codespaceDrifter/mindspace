#include "dense_layer.hpp"

DenseLayer::DenseLayer(int input, int output){
    this->name = "DenseLayer";
    this->weight = new Tensor (input, output);
    this->bias = new Tensor (1, output);
    this->weight->value_core->randomize();
    this->bias->value_core->randomize();
}

DenseLayer::~DenseLayer(){
    delete this->weight;
    delete this->bias;
}

Tensor* DenseLayer::forward(Tensor* input){
    Tensor* A = input->matmul(this->weight);
    Tensor* result = A->add(bias);
    return result;
}

std::vector<Tensor*> DenseLayer::self_parameters(){
    std::vector<Tensor*> result;
    result.push_back(this->weight);
    result.push_back(this->bias);
    return result;
}