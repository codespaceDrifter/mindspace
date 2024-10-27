#ifndef FFW_BLOCK_HPP
#define FFW_BLOCK_HPP


#include "dense_layer.hpp"
#include "relu_layer.hpp"
#include "dropout_layer.hpp"
#include "block.hpp"

class FFWBlock : public Block{
public:

FFWBlock(int input = 0, int hidden = 0, int output = 0, float dropout = 0.1);

Tensor* forward_(Tensor* input, Tensor* input2 = nullptr) override;
void init_members() override;

Block* dense_layer_1;
Block* relu_layer;
Block* dropout_layer;
Block* dense_layer_2;
};




#endif