#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include "block.hpp"

class ReluLayer : public Block {
public:

ReluLayer();

Tensor* forward_(Tensor* input, Tensor* input2 = nullptr) override;
void init_members() override;

};

#endif