#include "ffw_block.hpp"


FFWBlock::FFWBlock(int input, int hidden, int output, float dropout){
    this->type = "FFW_BLOCK";
    this->register_block<FFWBlock>();

    this->sub_blocks.push_back(new DenseLayer(input, hidden));
    this->sub_blocks.push_back(new ReluLayer());
    this->sub_blocks.push_back(new DropoutLayer(dropout));
    this->sub_blocks.push_back(new DenseLayer(hidden, output));
    this->init_members();
}

Tensor* FFWBlock::forward_(Tensor* input, Tensor* input2){

    Tensor * A = this->dense_layer_1->forward(input);
    Tensor * B = this->relu_layer->forward(A);
    Tensor * C = this->dropout_layer->forward(B);
    Tensor* result = this->dense_layer_2->forward(C);

    return result;
}

void FFWBlock::init_members() {
    assert (this->sub_blocks.size() == 4);

    this->dense_layer_1 = this->sub_blocks[0];
    this->relu_layer = this->sub_blocks[1];
    this->dropout_layer = this->sub_blocks[2];
    this->dense_layer_2 = this->sub_blocks[3];
}
