#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include "block.hpp"

class ReluLayer : public Block {
public:

ReluLayer();

Tensor* forward(Tensor* input) override;
Block* factory_create () override;
void init_members() override;

Tensor* zero;
};








#endif