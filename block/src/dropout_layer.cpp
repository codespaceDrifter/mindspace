#include "dropout_layer.hpp"

DropoutLayer::DropoutLayer(float percent){

    assert (percent >= 0 && percent <= 1);

    this->type = "DROPOUT_LAYER";
    this->register_block();

    this->percent = Tensor::make_a_num(percent);
    this->parameters.push_back(this->percent);
}

Tensor* DropoutLayer::forward(Tensor* input){

    if (this->training == false) return input;

    Tensor * rand = new Tensor (input->shape);
    rand->randomize(0,1);
    Tensor * mask = rand -> compare (this->percent);
    delete rand;
    Tensor* result = input->mul (mask);
    return result;
}

Block* DropoutLayer::factory_create (){
    return new DropoutLayer();
}

void DropoutLayer::init_members() {
    this->percent = this->parameters[0];
}
