#include "ffw_block.hpp"


FFWBlock::FFWBlock(int input, int output, float dropout){
    this->type = "FFW_BLOCK";
    this->register_block<FFWBlock>();

    this->dense_layer = new DenseLayer(input, output);
    this->relu_layer = new ReluLayer();
    this->dropout_layer = new DropoutLayer(dropout);
    this->sub_blocks.push_back(this->dense_layer);
    this->sub_blocks.push_back(this->relu_layer);
    this->sub_blocks.push_back(this->dropout_layer);
}

Tensor* FFWBlock::forward(Tensor* input){
    Tensor * A = this->dense_layer->forward(input);
    Tensor * B = this->relu_layer->forward(A);
    Tensor * result = this->dropout_layer->forward(B);

    return result;
}

void FFWBlock::init_members() {
    assert (this->sub_blocks.size() == 3);

    this->dense_layer = this->sub_blocks[0];
    this->relu_layer = this->sub_blocks[1];
    this->dropout_layer = this->sub_blocks[2];
}
