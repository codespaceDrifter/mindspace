#include "ffw_block.hpp"



FFWBlock::FFWBlock (int input, int output, float dropout){
    this->name = "FFWBlock";
    std::vector<int> all_dims;
    all_dims.insert(all_dims.begin(), input);
    all_dims.push_back(output);

    Block* dense_layer = new DenseLayer(input, output);
    Block* relu_layer = new ReluLayer ();
    Block* dropout_layer = new DropoutLayer(dropout);

    this->sub_blocks.push_back(dense_layer);
    this->sub_blocks.push_back(relu_layer);
    this->sub_blocks.push_back(dropout_layer);
}

void FFWBlock::delete_parameters(){}

Tensor* FFWBlock::forward(Tensor* input){
    Tensor* a = this->sub_blocks[0]->forward(input);
    Tensor* b = this->sub_blocks[1]->forward(a);
    Tensor* c = this->sub_blocks[2]->forward(b);
    return c;
}

std::vector<Tensor*> FFWBlock::self_parameters() {return {};};