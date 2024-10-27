#ifndef TRANSFORMER_MODEL_HPP
#define TRANSFORMER_MODEL_HPP

#include "encoder_block.hpp"
#include "decoder_block.hpp"
#include "embedding_layer.hpp"
#include "pos_embed_layer.hpp"
#include "dense_layer.hpp"
#include "softmax_layer.hpp"

class TransformerModel : public Block {
public:
    TransformerModel(int vocab_size = 0, int seq_len = 0, int embed_dim = 0, int num_heads = 1, int ffw_hidden = 0, 
                     int num_encoder_blocks = 0, int num_decoder_blocks = 0, 
                     float dropout_rate = 0.1);

    Tensor* forward_(Tensor* input, Tensor* input2) override;
    void init_members() override;

    Block* embedding;
    Block* pos_embed;
    std::vector<Block*> encoder_blocks;
    std::vector<Block*> decoder_blocks;
    Block* final_dense;
    Block* final_softmax;
};

#endif // TRANSFORMER_MODEL_HPP

