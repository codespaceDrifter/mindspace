#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "block.hpp"

class DropoutLayer : public Block{
public:

// input is number of weights, output is number of neurons
DropoutLayer(float percentage = 0.1);

void delete_parameters();
Tensor* forward(Tensor* input) override;
std::vector<Tensor*> self_parameters() override;

Tensor* percent;
};

#endif