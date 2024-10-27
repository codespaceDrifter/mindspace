#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "block.hpp"

class DenseLayer : public Block{

public:

// input is number of weights, output is number of neurons
DenseLayer(int input = 0, int output = 0);

Tensor* forward_(Tensor* input, Tensor* input2 = nullptr) override;
void init_members() override;

Tensor* weight;
Tensor* bias;

// parameter list:
// weight, bias
};

#endif