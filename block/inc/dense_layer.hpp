#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "block.hpp"

class DenseLayer : public Block{

public:

// input is number of weights, output is number of neurons
DenseLayer(int input, int output);

Tensor* forward(Tensor* input) override;
Block* factory_create () override;
void init_members() override;

Tensor* weight;
Tensor* bias;

// parameter list:
// weight, bias
};

#endif