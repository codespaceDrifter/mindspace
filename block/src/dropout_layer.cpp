#include "dropout_layer.hpp"

DropoutLayer::DropoutLayer(float percent){
    assert (percent >= 0 && percent <= 1);
    this->type = "DROPOUT_LAYER";
    this->register_block<DropoutLayer>();
    this->parameters.push_back(Tensor::make_a_num(percent));
    this->init_members();
}

Tensor* DropoutLayer::forward_(Tensor* input, Tensor* input2){

    if (Block::training == false) return input;

    Tensor * rand = new Tensor (input->shape);
    rand->randomize(0,1);
    Tensor * mask = rand -> compare (this->percent);
    delete rand;
    Tensor* result = input->mul (mask);
    return result;
}

void DropoutLayer::init_members() {
    this->percent = this->parameters[0];
}
