#ifndef BLOCK_TESTS_HPP
#define BLOCK_TESTS_HPP

#include "test.hpp"
#include "all_block_inc.hpp"

class BlockTest :public Test {
public:

void run_tests() override;

void dense_layer_test();

void relu_test();

void dropout_test();

void ffw_test();

void save_load_test();

void embedding_test();
};




#endif