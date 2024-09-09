#include "tensor_core.hpp"

//single core cpu with c++. no parallism
//maybe use c++ threads for batch level parallism later

TensorCore* CpuTensorCore::add(TensorCore* other){
    assert (this->element_op_viable(other));
    std::vector<int> result_shape = this->max_broadcast_shape(other);
    TensorCore* result = TensorCore::create_TensorCore(result_shape);
    for (int i = 0; i < result->data_size; ++i){
        std::vector<int> indices = result->f_s(i);
        result->idx(indices) = this->idx(indices) + other->idx(indices);
    }
    return result;
}


TensorCore* CpuTensorCore::mul(TensorCore* other){
    assert (this->element_op_viable(other));
    std::vector<int> result_shape = this->max_broadcast_shape(other);
    TensorCore* result = TensorCore::create_TensorCore(result_shape);
    for (int i = 0; i < result->data_size; ++i){
        std::vector<int> indices = result->f_s(i);
        result->idx(indices) = this->idx(indices) * other->idx(indices);
    }
    return result;
}

TensorCore* CpuTensorCore::reduce_sum(std::vector<int> result_shape){
    TensorCore* result = TensorCore::create_TensorCore(result_shape);
    for (int i = 0; i < this->data_size; ++i){
        std::vector<int> indices = this->f_s(i);
        result->idx(indices) += this->idx(indices);
    }
    return result;
}

TensorCore* CpuTensorCore::compare (TensorCore* other){
    assert (this->element_op_viable(other));
    std::vector<int> result_shape = this->max_broadcast_shape(other);
    TensorCore* result = TensorCore::create_TensorCore(result_shape);
    for (int i = 0; i < this->data_size; ++i){
        std::vector<int> indices = this->f_s(i);
        float cur_this = this->idx(indices);
        float cur_other = other->idx(indices);
        if (cur_this >= cur_other) result->idx(indices) = 1;
        else result->idx(indices) = 0;
    }
    return result;
}
