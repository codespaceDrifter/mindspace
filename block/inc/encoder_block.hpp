#ifndef ENCODER_BLOCK_HPP
#define ENCODER_BLOCK_HPP

#include "block.hpp"
#include "add_norm_layer.hpp"
#include "multihead_attention_block.hpp"
#include "ffw_block.hpp"

class EncoderBlock : public Block {
public:
    EncoderBlock(int seq_len = 0, int embed_dim = 0, int heads = 1, int ffw_hidden = 0, float dropout = 0.1);

    Tensor* forward_(Tensor* input, Tensor* input2 = nullptr) override;
    void init_members() override;

private:
    Block* mha_r;
    Block* ffw_r;
};

#endif // ENCODING_BLOCK_HPP

