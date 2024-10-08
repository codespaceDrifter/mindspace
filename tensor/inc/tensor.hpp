#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <map>
#include <algorithm>
#include <memory>
#include <utility>
#include <optional>
#include <cmath>
#include <set>
#include <fstream>
#include <cstdio>
#include <queue>


enum class Operation{
    None,
    View,
    Add,
    Minus,
    Mul,
    Div,
    Pow,
    ReduceSum,
    Max,
    Min,
    Matmul
};

class Tensor{

public:

Tensor (std::vector<int> shape, bool requires_grad = true);

template <typename... Indices>
Tensor (Indices... indices) : Tensor(std::vector<int> {indices ...}) {};

~Tensor ();

void compute_data_size ();
void data_init ();
void data_switch (float* new_data);
void compute_stride();

static Tensor* make_a_num (float val);

void randomize(float low = -0.5, float high = 0.5);
void fill(float value);
void arrange(int start = 0);

void print();
std::string to_string();


//this way of indexing is very slow. avoid using as much as possible
inline __attribute__((always_inline)) static std::vector<int> f_s (int idx){
    std::vector<int> result;
    int cur_group = idx;
    for (int i = this->shape.size() - 1; i >= 0; --i){
        result.insert(result.begin(),cur_group % this->shape[i]);
        cur_group = cur_group / this->shape[i];
    }
    return result;
}

inline __attribute__((always_inline)) float& idx (std::vector<int> indices_vec){
    indices_vec.erase(indices_vec.begin(), indices_vec.begin() + indices_vec.size() - this->shape.size());
    int data_idx = 0;
    for (int i = 0; i < indices_vec.size(); ++i){
        data_idx += indices_vec[i] * this->lazy_stride[i] + offset[i] * this->stride[i];
    }
    return this->data[data_idx];
}

template <typename... Indices>
inline __attribute__((always_inline)) float& idx (Indices... indices){
    std::vector<int> indices_vec = {indices...};
    return this->idx(indices_vec);
}

void transpose (int dim1 = -1, int dim2 = -2);
void squeeze (int dim);
void unsqueeze (int dim);
void shape_change (std::vector<int> new_shape);
void copy (Tensor* target);

Tensor* deep_broadcast(std::vector<int> target_shape);

//operations
Tensor* add (Tensor* other);
Tensor* minus (Tensor* other);
Tensor* mul (Tensor* other);
Tensor* div (Tensor* other);
Tensor* pow (Tensor* other);
Tensor* reduce_sum (std::vector<int> target_shape);
Tensor* compare (Tensor* other);
Tensor* max (Tensor* other);
Tensor* min (Tensor* other);
Tensor* matmul (Tensor* other);
Tensor* slice (std::vector<std::vector<int>> slices);

//other forms
Tensor* slice(std::initializer_list<std::initializer_list<int>> slices);
Tensor* reduce_sum (int target_dim);

//in place operations. does not allow broadcasting or backprop
void add_ (Tensor* other);
void minus_ (Tensor* other);
void mul_ (Tensor* other);
void div_ (Tensor* other);


//backprop and intermediate deletion
void init_grad ();
void topo_sort (std::set<Tensor*>& visited, std::vector<Tensor*>& sorted);
void backward_model (bool delete_intermediate = true);

//backprop operations
void backprop ();
void accumulate_grad(Tensor* grad_A, Tensor* grad_B);

void add_backprop();
void minus_backprop();
void mul_backprop();
void div_backprop();
void pow_backprop();
void reduce_sum_backprop();
void max_backprop();
void min_backprop();
void matmul_backprop();

//parameters

float* data;
int data_size;

std::vector<int> shape;
std::vector<int> stride;

bool requires_grad;
Tensor* grad;

Operation operation;
std::vector<Tensor*> operands;

//for view backprop
std::vector<int> view_info;

//saving and loading
void save (std::ofstream &out_file);
void load (std::ifstream &in_file);
int ID;
};

#endif //TENSOR_HPP