#include "pos_embed_layer.hpp"


PosEmbedLayer::PosEmbedLayer(int seq_len, int embed_dim){
    this->type = "POS_EMBED_LAYER";
    this->register_block<PosEmbedLayer>();

    Tensor* mask = new Tensor (seq_len,embed_dim);
    for (int i = 0; i < seq_len; ++i){
        for (int j = 0; j < embed_dim; ++j){
            int embed_pos = j/2;
            float temp = i / std::pow (10000, (static_cast<float>(embed_pos)/embed_dim));
            float val;
            if (j%2 == 0){
                val = std::sinf(temp);
            } else {
                val = std::cosf(temp);
            }
            mask ->idx (i,j) = val;
        }
    }
    this->parameters.push_back(mask);
    this->init_members();
}

/*
input:
any shape, embed_dim

(batch, seq_len, embed)

output:
(batch, seq_len, embed)
*/

Tensor* PosEmbedLayer::forward_(Tensor* input, Tensor* input2){
    assert (input->shape.size() >= 2);
    assert (input->shape[input->shape.size()-2] == this->mask->shape[0]);
    assert (input->shape[input->shape.size()-1] == this->mask->shape[1]);
    Tensor* result = input->add(this->mask);
    return result;
}

void PosEmbedLayer::init_members(){
    assert (this->parameters.size() == 1);
    this->mask = this->parameters[0];

    this->mask->requires_grad = false;
}