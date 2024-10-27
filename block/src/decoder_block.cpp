
#include "decoder_block.hpp"

DecoderBlock::DecoderBlock(int seq_len, int embed_dim, int num_heads, int ffw_hidden, float dropout) {
    this->type = "DECODER_BLOCK";
    this->register_block<DecoderBlock>();
    Block* self_attention = new MultiheadAttentionBlock(seq_len, embed_dim, num_heads, true);
    Block* cross_attention = new MultiheadAttentionBlock(seq_len, embed_dim, num_heads, false);
    Block* ffw = new FFWBlock(embed_dim, ffw_hidden, embed_dim, dropout);
    Block* self_attention_add_norm = new AddNormLayer(self_attention);
    Block* cross_attention_add_norm = new AddNormLayer(cross_attention);
    Block* ffw_add_norm = new AddNormLayer(ffw);
    this->sub_blocks.push_back(self_attention_add_norm);
    this->sub_blocks.push_back(cross_attention_add_norm);
    this->sub_blocks.push_back(ffw_add_norm);
    this->init_members();
}

Tensor* DecoderBlock::forward_(Tensor* input, Tensor* input2) {
    Tensor* self_attn_norm = this->self_attention_r->forward(input);
    Tensor* cross_attn_norm = this->cross_attention_r->forward(self_attn_norm, input2);
    Tensor* ffw_norm = this->ffw_r->forward(cross_attn_norm);
    return ffw_norm;
}

void DecoderBlock::init_members() {
    assert(this->sub_blocks.size() == 3);
    this->self_attention_r = this->sub_blocks[0];
    this->cross_attention_r = this->sub_blocks[1];
    this->ffw_r = this->sub_blocks[2];
}   
