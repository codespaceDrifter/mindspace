#include "tensor_core.hpp"

//single core cpu with c++. no parallism
//maybe use c++ threads for batch level parallism later

TensorCore* CpuTensorCore::add(TensorCore* other){
    assert (this->element_op_viable(other));
    std::vector<int> result_shape = this->max_broadcast_shape(other);
    TensorCore* result = TensorCore::createTensorCore(result_shape);
    for (int i = 0; i < result->data_size; ++i){
        std::vector<int> indices = result->f_s(i);
        result->idx(indices) = this->idx(indices) + other->idx(indices);
    }
    return result;
}


TensorCore* CpuTensorCore::mul(TensorCore* other){
    assert (this->element_op_viable(other));
    std::vector<int> result_shape = this->max_broadcast_shape(other);
    TensorCore* result = TensorCore::createTensorCore(result_shape);
    for (int i = 0; i < result->data_size; ++i){
        std::vector<int> indices = result->f_s(i);
        result->idx(indices) = this->idx(indices) * other->idx(indices);
    }
    return result;
}

TensorCore* CpuTensorCore::reduce_sum(std::vector<int> result_shape){
    TensorCore* result = TensorCore::createTensorCore(result_shape);
    for (int i = 0; i < this->data_size; ++i){
        std::vector<int> indices = this->f_s(i);
        result->idx(indices) += this->idx(indices);
    }
    return result;
}

TensorCore* CpuTensorCore::matmul(TensorCore* other){
    assert (this->matmul_viable(other));
    std::unique_ptr<TensorCore> A = this->unsqueeze(-2);
    std::unique_ptr<TensorCore> B = other->transpose()->unsqueeze(-3);

    TensorCore* C = A->mul(B);
    TensorCore* C_reduced = C->reduce_sum(-1);

    delete C;
    C_reduced->squeeze_(-1);
    return C_reduced;
}
