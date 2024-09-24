#ifndef TENSOR_TESTS_HPP
#define TENSOR_TESTS_HPP

#include "test.hpp"


class TensorTest : public Test{

public:

void run_tests() override;

void create_test();

void index_test();

void views_test();

void in_place_views_test();

void topo_sort_test();

void add_test();

void minus_test();

void mul_test();

void div_test();

void pow_test();

void reduce_sum_test();



void compare_test();

void max_test();

void min_test();

void matmul_test();

void in_place_ops_test();

void views_backprop_test();

};



#endif