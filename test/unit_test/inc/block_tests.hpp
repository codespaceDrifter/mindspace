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

void inference_test();

void save_load_test();

void embedding_test();

void pos_embed_test();

void add_norm_test();

void softmax_test();

void multiheaded_attention_test();

void encoder_block_test();

void decoder_block_test();

void transformer_model_test();
};




#endif