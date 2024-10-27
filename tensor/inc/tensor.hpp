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
#include <exception>


enum class Operation{
    None,
    Contigious,
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
void deep_equal (Tensor* other);

static Tensor* make_a_num (float val);

void randomize(float low = -0.5, float high = 0.5);
void fill(float value);
void arrange(float start = 0, float step = 1);



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


std::vector<int> max_broadcast_shape (Tensor* other) const;
static Tensor* vertical_stack (std::vector<Tensor*> tensor_vec);
Tensor* deep_broadcast(std::vector<int> target_shape);


//shape changes and views. acts on both this and grad tensor. does not own data
Tensor* create_view();
Tensor* slice(std::vector<std::vector<int>> slices);
Tensor* slice(std::initializer_list<std::initializer_list<int>> slices);
Tensor* transpose(int dim1 = -1, int dim2 = -2);
Tensor* squeeze(int dim);
Tensor* unsqueeze(int dim);
Tensor* shape_view (std::vector<int> target_shape);
Tensor* contiguous();

//in place
void squeeze_ (int dim);
void unsqueeze_ (int dim);


//backprop and intermediate deletion
void init_grad ();
void topo_sort (std::set<Tensor*>& visited, std::vector<Tensor*>& sorted);
void delete_intermediates();
void backward_model (bool delete_intermediate = true);

//operations
void graph_construction (Tensor* other, Tensor* result);

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
Tensor* reduce_sum (int target_dim);

//backprop operations
void backprop ();
void accumulate_grad(Tensor* grad_A, Tensor* grad_B);

void contigous_backprop();
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
std::vector<Tensor*> outputs;

bool owns_data;
Tensor* viewed;

//anchor handles the deletion of tensors in inference mode. anchor true tensors are deleted at the end of a complete inference rather than after a single step. 
bool anchor;
static std::vector<Tensor*> anchored_tensors;

void set_anchor_true();
void set_anchor_false();
static void clear_anchored();

//saving and loading
void save (std::ofstream &out_file);
void load (std::ifstream &in_file);
int ID;


//debugging tools
void print();
std::string to_string();
static std::string vec_str(std::vector<int> vec);
static std::string op_to_str (Operation op);
void print_tree(int tabs = 0);
static void ASDF(){std::cout<<"ASDF"<<std::endl;};


//float versions. operand resize is for deletion through delete_intermediates
Tensor* add (float other){Tensor* b = Tensor::make_a_num(other); Tensor* result = this->add(b); b->requires_grad = false; b->operands.resize(1); return result;};
Tensor* minus (float other){Tensor* b = Tensor::make_a_num(other); Tensor* result = this->minus(b); b->requires_grad = false; b->operands.resize(1); return result;};
Tensor* mul (float other){Tensor* b = Tensor::make_a_num(other); Tensor* result = this->mul(b); b->requires_grad = false; b->operands.resize(1); return result;};
Tensor* div (float other){Tensor* b = Tensor::make_a_num(other); Tensor* result = this->div(b); b->requires_grad = false; b->operands.resize(1); return result;};
Tensor* pow (float other){Tensor* b = Tensor::make_a_num(other); Tensor* result = this->pow(b); b->requires_grad = false; b->operands.resize(1); return result;};
Tensor* compare (float other){Tensor* b = Tensor::make_a_num(other); Tensor* result = this->compare(b); b->requires_grad = false; b->operands.resize(1); return result;};
Tensor* max (float other){Tensor* b = Tensor::make_a_num(other); Tensor* result = this->max(b); b->requires_grad = false; b->operands.resize(1); return result;};
Tensor* min (float other){Tensor* b = Tensor::make_a_num(other); Tensor* result = this->min(b); b->requires_grad = false; b->operands.resize(1); return result;};
void add_ (float other){Tensor* b = Tensor::make_a_num(other); this->add_(b); delete b;};
void minus_ (float other){Tensor* b = Tensor::make_a_num(other); this->minus_(b); delete b;};
void mul_ (float other){Tensor* b = Tensor::make_a_num(other); this->mul_(b); delete b;};
void div_ (float other){Tensor* b = Tensor::make_a_num(other); this->div_(b); delete b;};

};

#endif //TENSOR_HPP