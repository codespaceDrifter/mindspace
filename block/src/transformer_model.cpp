#include "transformer_model.hpp"

TransformerModel::TransformerModel(int vocab_size, int seq_len, int embed_dim, int num_heads, int ffw_hidden, 
                                   int num_encoder_blocks, int num_decoder_blocks, float dropout_rate){
    this->type = "TRANSFORMER_MODEL";
    this->register_block<TransformerModel>();

    //meta data
    Tensor* num_encoders = Tensor::make_a_num(num_encoder_blocks);
    Tensor* num_decoders = Tensor::make_a_num(num_decoder_blocks);
    this->parameters.push_back(num_encoders);
    this->parameters.push_back(num_decoders);   

    //embedding layers
    Block* embedding = new EmbeddingLayer(vocab_size, embed_dim);
    Block* pos_embed = new PosEmbedLayer(seq_len, embed_dim);

    // Encoder blocks
    std::vector<Block*> encoder_blocks;
    for (int i = 0; i < num_encoder_blocks; ++i) {
        encoder_blocks.push_back(new EncoderBlock(seq_len, embed_dim, num_heads, ffw_hidden, dropout_rate));
    }

    // Decoder blocks
    std::vector<Block*> decoder_blocks;
    for (int i = 0; i < num_decoder_blocks; ++i) {
        decoder_blocks.push_back(new DecoderBlock(seq_len, embed_dim, num_heads, ffw_hidden, dropout_rate));
    }

    // Final dense layer
    Block* final_dense = new DenseLayer(embed_dim, vocab_size);

    // Final softmax layer
    Block* final_softmax = new SoftmaxLayer();

    this->sub_blocks.push_back(embedding);
    this->sub_blocks.push_back(pos_embed);
    this->sub_blocks.insert(this->sub_blocks.end(), encoder_blocks.begin(), encoder_blocks.end());
    this->sub_blocks.insert(this->sub_blocks.end(), decoder_blocks.begin(), decoder_blocks.end());
    this->sub_blocks.push_back(final_dense);
    this->sub_blocks.push_back(final_softmax);
    this->init_members();
}

Tensor* TransformerModel::forward_(Tensor* input, Tensor* input2) {
    // Encoder
    Tensor* input_embed = this->embedding->forward(input);
    Tensor* input_pos_embed = this->pos_embed->forward(input_embed);
    
    Tensor* encoder_output = input_pos_embed;
    for (Block* encoder_block : this->encoder_blocks) {
        encoder_output = encoder_block->forward(encoder_output);
    }

    // Decoder
    Tensor* decoder_embed = this->embedding->forward(input2);
    Tensor* decoder_pos_embed = this->pos_embed->forward(decoder_embed);
    
    Tensor* decoder_output = decoder_pos_embed;
    for (Block* decoder_block : this->decoder_blocks) {
        decoder_output = decoder_block->forward(decoder_output, encoder_output);
    }

    // Final layers
    Tensor* dense_output = this->final_dense->forward(decoder_output);
    Tensor* softmax_output = this->final_softmax->forward(dense_output);

    return softmax_output;
}

void TransformerModel::init_members(){

    assert(this->parameters.size() == 2);   
    int num_encoders = static_cast<int> (this->parameters[0]->data[0]);
    int num_decoders = static_cast<int> (this->parameters[1]->data[0]);

    assert(this->sub_blocks.size() == 2 + num_encoders + num_decoders + 2);
    this->embedding = this->sub_blocks[0];
    this->pos_embed = this->sub_blocks[1];
    this->encoder_blocks = std::vector<Block*>(this->sub_blocks.begin() + 2, this->sub_blocks.begin() + 2 + num_encoders);
    this->decoder_blocks = std::vector<Block*>(this->sub_blocks.begin() + 2 + num_encoders, this->sub_blocks.begin() + 2 + num_encoders + num_decoders);
    this->final_dense = this->sub_blocks[this->sub_blocks.size() - 2];
    this->final_softmax = this->sub_blocks[this->sub_blocks.size() - 1];
}