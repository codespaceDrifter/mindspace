#ifndef MULTIHEAD_ATTENTION_BLOCK
#define MULTIHEAD_ATTENTION_BLOCK

#include "block.hpp"
#include "softmax_layer.hpp"
#include "dense_layer.hpp"

class MultiheadAttentionBlock : public Block {
public:
MultiheadAttentionBlock (int seq_len = 0, int embed_dim = 0, int heads = 1, bool mask = false);

Tensor* forward_(Tensor* input, Tensor* input2 = nullptr) override;
void init_members() override;

Tensor* heads;
Tensor* need_mask;
Tensor* causal_mask;

Block* query_dense;
Block* key_dense;
Block* value_dense;
Block* output_dense;

Block* softmax;
};




#endif