//could have multiple dense layers

#ifndef FFW_BLOCK_HPP
#define FFW_BLOCK_HPP

#include "dense_layer.hpp"
#include "relu_layer.hpp"
#include "dropout_layer.hpp"


//dense, relu, dropout

class FFWBlock : public Block{
public:

FFWBlock (int input, int output, float dropout);

void delete_parameters();
Tensor* forward(Tensor* input) override;
std::vector<Tensor*> self_parameters() override;
};



#endif