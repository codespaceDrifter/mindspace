#ifndef DECODER_BLOCK_HPP
#define DECODER_BLOCK_HPP

#include "block.hpp"
#include "encoder_block.hpp"

class DecoderBlock : public Block {
public:
    DecoderBlock(int seq_len = 0, int embed_dim = 0, int num_heads = 1, int ffw_hidden = 0, float dropout = 0.1);
    virtual Tensor* forward_(Tensor* input, Tensor* input2 = nullptr) override;
    virtual void init_members() override;

    Block* self_attention_r;
    Block* cross_attention_r;
    Block* ffw_r;
};

#endif // DECODER_BLOCK_HPP

