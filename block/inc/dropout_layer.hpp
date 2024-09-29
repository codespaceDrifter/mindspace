#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "block.hpp"

class DropoutLayer : public Block {

public:

DropoutLayer (float percent = 0.1);

Tensor* forward(Tensor* input) override;
Block* factory_create () override;
void init_members() override;

Tensor* percent;
};





#endif