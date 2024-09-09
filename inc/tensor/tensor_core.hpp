#ifndef TENSOR_CORE_HPP
#define TENSOR_CORE_HPP

#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include <optional>

//common class defined in tensor_core.cpp. specific vendors for parallelism defined in engine folder



class TensorCore{

public:

template <typename... Indices>
static TensorCore* create_TensorCore(Indices... indices){
    std::vector<int> indices_vec = {indices...};
    return TensorCore::create_TensorCore(indices_vec);
}     

//creates a platform specific tensorcore.
static TensorCore* create_TensorCore(std::vector<int> shape);


TensorCore(): TensorCore (std::vector<int>{1,1}){};

TensorCore (std::vector<int> shape);

template <typename... Indices>
TensorCore (Indices... indices) : TensorCore(std::vector<int> {indices ...}){};

TensorCore (std::vector<float> data, std::vector<int> shape);

~TensorCore();


void data_resize(int new_size);

void data_switch (float* new_data, int new_size = -1);

//stride use lazy broadcasting, so strides for where shape 1 is 0.
void contiguous();

static TensorCore* make_a_num (float num);

void randomize(float low = -0.5, float high = 0.5);
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
        data_idx += (indices_vec[i] + offset[i])* this->stride[i];
    }
    return this->data[data_idx];
}

template <typename... Indices>
inline __attribute__((always_inline)) float& idx (Indices... indices){
    std::vector<int> indices_vec = {indices...};
    return this->idx(indices_vec);
}


//shallow equal holds the same data ptr. does not call shape update to make data contiguous
std::unique_ptr<TensorCore> create_shallow_equal ();
std::unique_ptr<TensorCore> create_deep_equal ();


using Slice = std::optional<std::pair<int, int>>; 
std::unique_ptr<TensorCore> slice(std::vector<Slice> slices);
void slice_ (std::vector<Slice> slices);

static TensorCore* vertical_stack (std::vector<TensorCore*> tensor_vec);

std::unique_ptr<TensorCore> transpose(int dim1 = -1, int dim2 = -2);
void transpose_ (int dim1 = -1, int dim2 = -2);

std::unique_ptr<TensorCore> squeeze(int dim);
void squeeze_ (int dim);

std::unique_ptr<TensorCore> unsqueeze(int dim);
void unsqueeze_ (int dim);

TensorCore* toggle_bits();
std::vector<int> max_broadcast_shape (TensorCore* other) const;
bool element_op_viable(TensorCore* other) const;
bool matmul_viable (TensorCore* other) const;

//operations need to be parallized in backends in engine folder
virtual TensorCore* add (TensorCore* other) = 0;
virtual TensorCore* mul (TensorCore* other) = 0;
virtual TensorCore* reduce_sum (std::vector<int> result_shape) = 0;
virtual TensorCore* compare (TensorCore* other) = 0;

//operations that are NOT optimized. they use parallized operations above, but either lead to TOO high memory or are NOT optimized for speed for specific hardware
//might fix later
TensorCore* matmul (TensorCore* other);
TensorCore* max (TensorCore* other);
TensorCore* min (TensorCore* other);

//parameters
float* data;
int data_size;
bool owns_data;

std::vector<int> shape;
std::vector<int> offset;
std::vector<int> stride;


// operation other forms

TensorCore* add (int val);
TensorCore* mul (int val);
TensorCore* reduce_sum (int idx);

//unique ptr versions
std::vector<int> max_broadcast_shape (std::unique_ptr<TensorCore>& other) const {return this->max_broadcast_shape(other.get());};
bool element_op_viable(std::unique_ptr<TensorCore>& other) const {return this->element_op_viable(other.get());};
bool matmul_viable (std::unique_ptr<TensorCore>& other) const {return this->matmul_viable(other.get());};
TensorCore* add (std::unique_ptr<TensorCore>& other) {return this->add(other.get());};
TensorCore* mul (std::unique_ptr<TensorCore>& other) {return this->mul(other.get());};
TensorCore* matmul (std::unique_ptr<TensorCore>& other) {return this->matmul(other.get());};
TensorCore* compare (std::unique_ptr<TensorCore>& other) {return this->compare(other.get());};
TensorCore* max (std::unique_ptr<TensorCore>& other) {return this->max(other.get());};
TensorCore* min (std::unique_ptr<TensorCore>& other) {return this->min(other.get());};
};

class CpuTensorCore : public TensorCore{
public:
using TensorCore::TensorCore;
TensorCore* add (TensorCore* other) override ;
TensorCore* mul (TensorCore* other) override;
TensorCore* reduce_sum (std::vector<int> result_shape) override;
TensorCore* compare (TensorCore* other) override;
};

//add OpenclTensorCore, CudaTensorCore, MetalTensorCore, etc.
#endif