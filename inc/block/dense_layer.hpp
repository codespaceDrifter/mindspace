#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "block.hpp"

class DenseLayer : public Block{

public:

// input is number of weights, output is number of neurons
DenseLayer(int input, int output);


void delete_parameters();
Tensor* forward(Tensor* input) override;
std::vector<Tensor*> self_parameters() override;

Tensor* weight;
Tensor* bias;
};

#endif