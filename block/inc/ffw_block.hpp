#ifndef FFW_BLOCK_HPP
#define FFW_BLOCK_HPP


#include "dense_layer.hpp"
#include "relu_layer.hpp"
#include "dropout_layer.hpp"
#include "block.hpp"

class FFWBlock : public Block{
public:

FFWBlock(int input, int output, float dropout);

Tensor* forward(Tensor* input) override;
Block* factory_create () override;
void init_members() override;

Block* dense_layer;
Block* relu_layer;
Block* dropout_layer;
};




#endif