#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "block.hpp"

class DropoutLayer : public Block{
public:

// input is number of weights, output is number of neurons
DropoutLayer(std::vector<int>input_shape, float percentage = 0.1);

~DropoutLayer() override;

Tensor* forward(Tensor* input) override;

Tensor* mask;
};

#endif