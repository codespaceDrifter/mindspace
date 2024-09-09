#ifndef BLOCK_HPP
#define BLOCK_HPP


#include "tensor.hpp"

class Block {
public:
virtual ~Block() {}

virtual Tensor* forward(Tensor* input) = 0;

virtual std::vector<Tensor*> self_parameters() = 0;

std::vector<Tensor*> sub_blocks_parameters(){
    std::vector<Tensor*> result;
    for (Block* sub_block : this->sub_blocks){
        std::vector<Tensor*> sub_block_parameters = sub_block->parameters();
        result.insert(result.end(), sub_block_parameters.begin(),sub_block_parameters.end());
    }
    return result;
}

std::vector<Tensor*> parameters(){
    std::vector<Tensor*> result;
    std::vector<Tensor*> self = this->self_parameters();
    std::vector<Tensor*> component = this->sub_blocks_parameters();
    result.insert(result.end(), self.begin(), self.end());
    result.insert(result.end(), component.begin(), component.end());
    return result;
}

void set_mode(bool training) {this->training = training;}

std::vector<Block*> sub_blocks;
bool training = true; 
std::string name;
};

#endif

/*
save load format


Blocks: Blocks have two parts. parameter tensor number and subblocks
the size of meta data bytes is recorded at the start, so program knows when to start loading tensors
i.e.

FFW BLOCK
0
    DenseBlock
    2

    ReluBlock
    1

    DropoutLayer
    1


TENSORS: two lines. tensors seperated by empty lines
first line, shape. a vector of ints.
second line, float, a array of floats
*/