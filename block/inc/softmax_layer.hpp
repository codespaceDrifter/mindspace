#ifndef SOFTMAX_LAYER
#define SOFTMAX_LAYER

#include "block.hpp"

class SoftmaxLayer : public Block {
public:

SoftmaxLayer();

Tensor* forward_(Tensor* input, Tensor* input2 = nullptr) override;
void init_members() override;

};

#endif