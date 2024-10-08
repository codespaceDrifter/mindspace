#ifndef EMBEDDING_LAYER_HPP
#define EMBEDDING_LAYER_HPP

#include "block.hpp"


/*
input needs to be encoded as ints from 0 to vocab_size

(batch, seq_len) -> (batch, seq_len, embedding)

this is slow because have to index the whole thing sequentially
*/

class EmbeddingLayer: public Block {
public:
EmbeddingLayer (int vocab_size = 0, int embed_dim = 0);

Tensor* forward(Tensor* input) override;
void init_members() override;

Tensor* embedding;
};



#endif