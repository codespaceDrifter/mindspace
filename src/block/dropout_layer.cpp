#include "dropout_layer.hpp"

DropoutLayer::DropoutLayer(float percentage){
    this->name = "DropoutLayer";
    this->percent->value_core.reset(TensorCore::make_a_num(percentage));
}

void DropoutLayer::delete_parameters(){
    delete this->percent;
}

Tensor* DropoutLayer::forward(Tensor* input){

    std::vector<int> input_shape = input->value_core->shape;

    TensorCore* random_tensorcore = TensorCore::create_TensorCore(input_shape);
    random_tensorcore->randomize(0,1);
    TensorCore* mask_tensorcore = random_tensorcore->min(this->percent->value_core);
    delete random_tensorcore;

    Tensor* mask = new Tensor(input_shape);
    mask->intermediate = true;
    mask->value_core.reset(mask_tensorcore);

    Tensor* result = input -> mul (mask);
    return result;
}

//this is for saving and loading, not updating
std::vector<Tensor*> DropoutLayer::self_parameters() {
    std::vector<Tensor*> result;
    result.push_back(this->percent);
    return {result};
}