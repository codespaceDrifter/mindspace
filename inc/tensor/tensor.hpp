#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "tensor_core.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <map>
#include <memory>


enum class operationType{
    None,
    Add,
    Mul,
    Matmul,
    Max
};

struct Operation{
    operationType type;
    std::vector<int> additional_values;
    Operation (operationType type = operationType::None){
        this->type = type;
    }
};

//a tensor has: a value tensorcore and a grad tensorcore
//a tensor's shape must not change once its set
class Tensor{

public:

Tensor (std::vector<int> shape, bool intermediate);

template <typename... Indices>
Tensor (Indices... indices) : Tensor(std::vector<int> {indices ...}, false){};

void init_grad ();
void one_grad ();

//inital data tensor and weight times weight tensors need intermediate to set true manually, not with this
void set_intermediate();

//operations
void backprop ();

Tensor* add (Tensor* other);
void add_backprop();

Tensor* mul (Tensor* other);
void mul_backprop();

Tensor* matmul (Tensor* other);
void matmul_backprop();

//order matters. this needs to be bigger than other in all dims
Tensor* max (Tensor* other);
void max_backprop();

//member variables
std::unique_ptr<TensorCore> value_core;
std::unique_ptr<TensorCore> grad_core;

//intermediate tensors are deleted after each forward backward pass as opposed to weights
bool intermediate;

std::vector<Tensor*> operands;
Operation op;
};

#endif //TENSOR_HPP