#include "relu_layer.hpp"


ReluLayer::ReluLayer(){
    this->name = "ReluLayer";
    this->zero = new Tensor(1);
    this->zero->value_core->fill(0);
    this->zero->intermediate = true;
}

void ReluLayer::delete_parameters(){
    delete this->zero;
}

Tensor* ReluLayer::forward(Tensor* input){
    Tensor* result = input->max(this->zero);
    return result;
}


std::vector<Tensor*> ReluLayer::self_parameters(){
    return {};
}
