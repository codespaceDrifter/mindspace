#include "encoder_block.hpp"

EncoderBlock::EncoderBlock(int seq_len, int embed_dim, int heads, int ffw_hidden, float dropout) {
    this->type = "ENCODER_BLOCK";
    this->register_block<EncoderBlock>();

    Block* mha = new MultiheadAttentionBlock(seq_len, embed_dim, heads,false);
    Block* ffw = new FFWBlock(embed_dim, ffw_hidden, embed_dim, dropout);
    Block* mha_add_norm = new AddNormLayer(mha);    
    Block* ffw_add_norm = new AddNormLayer(ffw);
    this->sub_blocks.push_back(mha_add_norm);
    this->sub_blocks.push_back(ffw_add_norm);

    this->init_members();
}

Tensor* EncoderBlock::forward_(Tensor* input, Tensor* input2) {
    Tensor* mha_norm = this->mha_r->forward(input);
    Tensor* result = this->ffw_r->forward(mha_norm);
    return result;
}

void EncoderBlock::init_members() {
    assert(this->sub_blocks.size() == 2);

    this->mha_r = this->sub_blocks[0];
    this->ffw_r = this->sub_blocks[1];
}
