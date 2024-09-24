#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP


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


// implementations defined in engine folder

// do not pass Tensor* output with existing values. either pass unitialized pointers or in place operations, or it will be cause memory leak.


class Tensor;

namespace Op{
//only cpu and cuda implementations for now
enum class platforms{
    Cpu,
    Cuda,
    Opencl,
    Metal
};

//implement actual logic later
platforms get_platforms();


//parallel ones
void add (Tensor* A, Tensor*B, Tensor*& output);
void add_cpu(Tensor* A, Tensor*B, Tensor* output);

void minus (Tensor* A, Tensor*B, Tensor*& output);
void minus_cpu(Tensor* A, Tensor*B, Tensor* output);

void mul (Tensor* A, Tensor*B, Tensor*& output);
void mul_cpu(Tensor* A, Tensor*B, Tensor* output);

void div (Tensor* A, Tensor*B, Tensor*& output);
void div_cpu (Tensor* A, Tensor*B, Tensor* output);

void pow (Tensor*A, Tensor*B, Tensor*& output);
void pow_cpu (Tensor* A, Tensor*B, Tensor* output);

void log (Tensor*A, Tensor*B, Tensor*& output);
void log_cpu (Tensor*A, Tensor*B, Tensor* output);

void reduce_sum (Tensor* A, std::vector<int> target_shape, Tensor*& output);
void reduce_sum_cpu (Tensor* A, std::vector<int> target_shape, Tensor* output);

void compare (Tensor*A, Tensor*B, Tensor*& output);
void compare_cpu (Tensor*A, Tensor*B, Tensor* output);

void max (Tensor* A, Tensor* B, Tensor*& output);
void max_cpu (Tensor* A, Tensor* B, Tensor* output);

void min (Tensor* A, Tensor* B, Tensor*& output);
void min_cpu (Tensor* A, Tensor* B, Tensor* output);


//unoptimized ones that uses the parallel ones.
void matmul (Tensor* A, Tensor*B, Tensor*& output);

//operations other forms
void add (Tensor* A, float B, Tensor*& output);
void minus (Tensor* A, float B, Tensor*& output);
void minus (float A, Tensor* B, Tensor*& output);
void mul (Tensor* A, float B, Tensor*& output);
void div (Tensor* A, float B, Tensor*& output);
void pow (Tensor* A, float B, Tensor*& output);
void log (Tensor*A, float B, Tensor*& output);
void compare (Tensor* A, float B, Tensor*& output);
void max (Tensor* A, float B, Tensor*& output);
void min (Tensor* A, float B, Tensor*& output);

void reduce_sum (Tensor* A, int target_dim, Tensor*& output);


//create output tensor if not in place
void element_create (Tensor*A, Tensor*B, Tensor*& output);

//checking viability
bool appro_equal (float a, float b, float episolon = 1e-5);
bool element_op_viable(Tensor*A, Tensor*B, Tensor* output);
bool div_viable (Tensor* B);
bool log_viable (Tensor* A, Tensor* B);
bool matmul_viable (Tensor*A, Tensor*B);
}





#endif