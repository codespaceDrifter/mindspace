#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include "block.hpp"

class ReluLayer : public Block{
public:

ReluLayer();

~ReluLayer() override;
Tensor* forward(Tensor* input) override;
std::vector<Tensor*> self_parameters() override;

Tensor* zero;
};

#endif