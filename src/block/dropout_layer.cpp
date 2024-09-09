#include "dropout_layer.hpp"

DropoutLayer::DropoutLayer(std::vector<int> input_shape, float percentage){
    this->name = "DropoutLayer";
    this->mask = new Tensor(input_shape);
    this->mask->intermediate = true;

    TensorCore* random_tensorcore = TensorCore::create_TensorCore(input_shape);
    random_tensorcore->randomize(0,1);
    TensorCore* percentage_tensorcore = TensorCore::make_a_num(percentage);
    TensorCore* mask_tensorcore = random_tensorcore->min(percentage_tensorcore);
    delete random_tensorcore;
    delete percentage_tensorcore;
    this->mask->value_core.reset(mask_tensorcore);
}

DropoutLayer::~DropoutLayer(){
    delete this->mask;
}

Tensor* DropoutLayer::forward(Tensor* input){
    Tensor* result = input -> mul (this->mask);
    return result;
}