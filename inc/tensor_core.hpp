#ifndef TENSOR_CORE_HPP
#define TENSOR_CORE_HPP

#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <map>
#include <algorithm>
#include <memory>

//common class defined in tensor_core.cpp. specific vendors for parallelism defined in [platform]_tensor_core.cpp

class TensorCore{

public:

template <typename... Indices>
static TensorCore* createTensorCore(Indices... indices){
    std::vector<int> indices_vec = {indices...};
    return TensorCore::createTensorCore(indices_vec);
}     

//creates a platform specific tensorcore.
static TensorCore* createTensorCore(std::vector<int> shape);

TensorCore(): TensorCore (std::vector<int>{1,1}){};

TensorCore (std::vector<int> shape);

template <typename... Indices>
TensorCore (Indices... indices) : TensorCore(std::vector<int> {indices ...}){};

TensorCore (std::vector<float> data, std::vector<int> shape);

~TensorCore();


void data_resize(int new_size);

void data_switch (float* new_data, int new_size = -1);

//stride use lazy broadcasting, so strides for where shape 1 is 0.
void shape_update();

void randomize(float low = -1, float high = 1);
void fill(float value);
void arrange(int start = 0);

//prints shape and data
void print();

//makes tensor string in format of each dimension being a vertical visual diff
std::string to_string();

// converts a flat index to a shape index
inline __attribute__((always_inline)) static std::vector<int> shape_to_indices(const std::vector<int>& shape, int idx){
    std::vector<int> result;
    int cur_group = idx;
    for (int i = shape.size() - 1; i >= 0; --i){
        result.insert(result.begin(),cur_group % shape[i]);
        cur_group = cur_group / shape[i];
    }
    return result;
}

inline __attribute__((always_inline)) std::vector<int> f_s (int data_idx){
    return TensorCore::shape_to_indices(this->shape,data_idx);
}

inline __attribute__((always_inline)) float& idx (std::vector<int> indices_vec){

    indices_vec.erase(indices_vec.begin(), indices_vec.begin() + indices_vec.size() - this->shape.size());
    int data_idx = 0;
    for (int i = 0; i < indices_vec.size(); ++i){
        data_idx += (indices_vec[i])* this->stride[i];
    }
    return this->data[data_idx];
}

template <typename... Indices>
inline __attribute__((always_inline)) float& idx (Indices... indices){
    std::vector<int> indices_vec = {indices...};
    return this->idx(indices_vec);
}


//shallow equal holds the same data ptr
std::unique_ptr<TensorCore> create_shallow_equal ();
std::unique_ptr<TensorCore> create_deep_equal ();

//not contiguous. can use squeeze and unsqueeze after tho
std::unique_ptr<TensorCore> transpose(int dim1 = -1, int dim2 = -2);
// contiguous
void transpose_ (int dim1 = -1, int dim2 = -2);

std::unique_ptr<TensorCore> squeeze(int dim);
void squeeze_ (int dim);

std::unique_ptr<TensorCore> unsqueeze(int dim);
void unsqueeze_ (int dim);

std::vector<int> max_broadcast_shape (TensorCore* other) const;


bool element_op_viable(TensorCore* other) const;
bool matmul_viable (TensorCore* other) const;

TensorCore* reduce_sum (int idx){

    if (idx < 0) idx = this->shape.size() + idx;
    assert (this->shape.size() > idx && idx > 0);
    std::vector<int> result_shape = this->shape;
    result_shape[idx] = 1;
    return this->reduce_sum(result_shape);
}

//currently all operations assume a (batch, row, col) operates on (row, col) shape. operates on last two dimensions and broadcasts into batch
virtual TensorCore* add (TensorCore* other) = 0;
virtual TensorCore* mul (TensorCore* other) = 0;
virtual TensorCore* reduce_sum (std::vector<int> result_shape) = 0;
virtual TensorCore* matmul (TensorCore* other) = 0;



//unique ptr versions
std::vector<int> max_broadcast_shape (std::unique_ptr<TensorCore>& other) const {return this->max_broadcast_shape(other.get());};
bool element_op_viable(std::unique_ptr<TensorCore>& other) const {return this->element_op_viable(other.get());};
bool matmul_viable (std::unique_ptr<TensorCore>& other) const {return this->matmul_viable(other.get());};
TensorCore* add (std::unique_ptr<TensorCore>& other) {return this->add(other.get());};
TensorCore* mul (std::unique_ptr<TensorCore>& other) {return this->mul(other.get());};
TensorCore* matmul (std::unique_ptr<TensorCore>& other) {return this->matmul(other.get());};

float* data;
int data_size;
bool owns_data;

std::vector<int> shape;
std::vector<int> stride;

};

class CpuTensorCore : public TensorCore{
public:
using TensorCore::TensorCore;
TensorCore* add (TensorCore* other) override ;
TensorCore* mul (TensorCore* other) override;
TensorCore* matmul (TensorCore* other) override;
TensorCore* reduce_sum (std::vector<int> result_shape) override;
};

//add OpenclTensorCore, CudaTensorCore, MetalTensorCore, etc.
#endif