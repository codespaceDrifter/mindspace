#include "embedding_layer.hpp"


EmbeddingLayer::EmbeddingLayer(int vocab_size, int embed_dim){
    this->type = "EMBEDDING_LAYER";
    this->register_block<EmbeddingLayer>();
    this->parameters.push_back(new Tensor (vocab_size, embed_dim));
    this->init_members();
}

Tensor* EmbeddingLayer::forward_(Tensor* input, Tensor* input2){
    std::vector<int> result_shape (input->shape);
    result_shape.push_back(this->embedding->shape.back());

    Tensor* result = new Tensor (result_shape);

    for (int i = 0; i < input->shape_size; ++i){
        std::vector<int> input_indices = input->f_s(i);
        int token_id = static_cast<int> (input->idx(input_indices));

        std::vector<int> result_indices (input_indices);
        result_indices.push_back(0);
        for (int j = 0; j < this->embedding->shape.back(); ++j){
            result_indices.back() = j;
            result->idx (result_indices) = this->embedding->idx(token_id, j);
        }
    }
    return result;
}


void EmbeddingLayer::init_members(){
    this->embedding = this->parameters[0];
}