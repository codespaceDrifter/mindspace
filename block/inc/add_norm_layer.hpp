
#ifndef ADD_NORM_LAYER
#define ADD_NORM_LAYER


#include "block.hpp"

class AddNormLayer : public Block {
public:

AddNormLayer(Block* compoment = nullptr);

Tensor* forward_(Tensor* input, Tensor* input2 = nullptr) override;
void init_members() override;

Block* component;
};

#endif