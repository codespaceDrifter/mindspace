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
    Matmul,
    reduce_sum,
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

Tensor (std::vector<int> shape);

template <typename... Indices>
Tensor (Indices... indices) : Tensor(std::vector<int> {indices ...}){};

void init_grad ();
void one_grad ();

//operations
void backprop ();

Tensor* add (Tensor* other);
void add_backprop();

Tensor* matmul (Tensor* other);
void matmul_backprop();

//shape needs to contiguous at least, data do not. add an offset to the value_core of the result
//Tensor* slice ()


//member variables
std::unique_ptr<TensorCore> value_core;
std::unique_ptr<TensorCore> grad_core;

std::vector<Tensor*> grad_fn;
Operation op;
};

#endif //TENSOR_HPP