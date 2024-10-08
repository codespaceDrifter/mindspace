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

void data_init (int new_size);
void data_switch (float* new_data);
void compute_stride ();
void compute_lazy_stride ();
void offset_init();
void compute_shape_size();

void tensor_init();
void contiguous();

void view_other (Tensor* other);
void deep_equal (Tensor* other);

static Tensor* make_a_num (float val);

void randomize(float low = -0.5, float high = 0.5);
void fill(float value);
void arrange(int start = 0);

void print();
std::string to_string();
static void ASDF(){std::cout<<"ASDF"<<std::endl;};
static std::string vec_str(std::vector<int> vec);



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
    return Tensor::shape_to_indices(this->shape,data_idx);
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


std::unique_ptr<Tensor> create_view();
std::vector<int> max_broadcast_shape (Tensor* other) const;
static Tensor* vertical_stack (std::vector<Tensor*> tensor_vec);
Tensor* deep_broadcast(std::vector<int> target_shape);


//shape changes and views. acts on both this and grad tensor. does not own data
std::unique_ptr<Tensor> slice(std::vector<std::vector<int>> slices);
std::unique_ptr<Tensor> transpose(int dim1 = -1, int dim2 = -2);
std::unique_ptr<Tensor> squeeze(int dim);
std::unique_ptr<Tensor> unsqueeze(int dim);


//in place shape changes and views.
void slice_ (std::vector<std::vector<int>> slices);
void transpose_ (int dim1 = -1, int dim2 = -2);
void squeeze_ (int dim);
void unsqueeze_ (int dim);

//backprop and intermediate deletion
void init_grad ();
void topo_sort (std::set<Tensor*>& visited, std::vector<Tensor*>& sorted);
void delete_intermediates();
void backward_model (bool delete_intermediate = true);

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

//in place operations. does not allow broadcasting or backprop
void add_ (Tensor* other);
void minus_ (Tensor* other);
void mul_ (Tensor* other);
void div_ (Tensor* other);

//operations other forms
std::unique_ptr<Tensor> slice(std::initializer_list<std::initializer_list<int>> slices);
void slice_ (std::initializer_list<std::initializer_list<int>> slices);
Tensor* slice_r(std::initializer_list<std::initializer_list<int>> slices);
Tensor* reduce_sum (int target_dim);

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
std::vector<int> lazy_stride;
std::vector<int> offset;
int shape_size;

bool requires_grad;
Tensor* grad;

Operation operation;
std::vector<Tensor*> operands;

bool owns_data;
Tensor* viewed;

//raw ptr versions
Tensor* slice_r(std::vector<std::vector<int>> slices) {std::unique_ptr<Tensor> temp = this->slice(slices); return temp.release();};
Tensor* transpose_r(int dim1 = -1, int dim2 = -2) {std::unique_ptr<Tensor> temp = this->transpose(dim1, dim2); return temp.release();};
Tensor* squeeze_r(int dim) {std::unique_ptr<Tensor> temp = this->squeeze(dim); return temp.release();};
Tensor* unsqueeze_r(int dim) {std::unique_ptr<Tensor> temp = this->unsqueeze(dim); return temp.release();};

//unique ptr versions
void view_other (std::unique_ptr<Tensor>& other){view_other(other.get());};
void deep_equal (std::unique_ptr<Tensor>& other){deep_equal(other.get());};
Tensor* add (std::unique_ptr<Tensor>& other) {return this->add(other.get());};
Tensor* minus (std::unique_ptr<Tensor>& other) {return this->minus(other.get());};
Tensor* mul (std::unique_ptr<Tensor>& other) {return this->mul(other.get());};
Tensor* div (std::unique_ptr<Tensor>& other) {return this->div(other.get());};
Tensor* pow (std::unique_ptr<Tensor>& other) {return this->pow(other.get());};
Tensor* compare (std::unique_ptr<Tensor>& other) {return this->compare(other.get());};
Tensor* max (std::unique_ptr<Tensor>& other) {return this->max(other.get());};
Tensor* min (std::unique_ptr<Tensor>& other) {return this->min(other.get());};
Tensor* matmul (std::unique_ptr<Tensor>& other) {return this->matmul(other.get());};

//saving and loading
void save (std::ofstream &out_file);
void load (std::ifstream &in_file);
int ID;
};

#endif //TENSOR_HPP