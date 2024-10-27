#include "multihead_attention_block.hpp"

MultiheadAttentionBlock::MultiheadAttentionBlock(int seq_len, int embed_dim, int heads, bool mask){
    this->type = "MULTIHEAD_ATTENTION_BLOCK";
    this->register_block<MultiheadAttentionBlock>();

    assert (embed_dim % heads == 0);

    this->parameters.push_back(Tensor::make_a_num(static_cast<float>(heads)));
    this->parameters.push_back(Tensor::make_a_num(static_cast<float>(mask)));
    Tensor* causal_mask = new Tensor (seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i){
        for (int j = 0; j < seq_len; ++j){
            if (j>i) causal_mask->idx(i,j) = -60000.0;
        }
    }
    causal_mask->requires_grad = false;
    this->parameters.push_back(causal_mask);

    this->sub_blocks.push_back(new DenseLayer (embed_dim,embed_dim));
    this->sub_blocks.push_back(new DenseLayer (embed_dim,embed_dim));
    this->sub_blocks.push_back(new DenseLayer (embed_dim,embed_dim));
    this->sub_blocks.push_back(new DenseLayer (embed_dim,embed_dim));
    this->sub_blocks.push_back(new SoftmaxLayer());
    this->init_members();
}

Tensor* MultiheadAttentionBlock::forward_(Tensor* input, Tensor* input2){
    //input: (batch, seq, embed)
    Tensor* key_value;
    if (input2 == nullptr){
        key_value = input;
    }else {
        key_value = input2;
    }


    // Q,K,V: (batch, seq, embed)
    Tensor* Q = this->query_dense->forward(input);
    Tensor* K = this->key_dense->forward(key_value);
    Tensor* V = this->value_dense->forward(key_value);

    // Q,K,V (batch, seq, head, d_k)
    int heads = this->heads->data[0];
    int embed = input->shape[input->shape.size()-1];
    int d_k = embed / heads;
    std::vector<int> multihead_shape (Q->shape);
    multihead_shape.pop_back();
    multihead_shape.push_back(heads);
    multihead_shape.push_back(d_k);
    Tensor* Q_v = Q->shape_view(multihead_shape);
    Tensor* K_v = K->shape_view(multihead_shape);
    Tensor* V_v = V->shape_view(multihead_shape);
    // Q,K,V (batch, head, seq, d_k)
    Tensor* Q_h = Q_v->transpose(-2,-3);
    Tensor* K_h = K_v->transpose(-2,-3);
    Tensor* V_h = V_v->transpose(-2,-3);
    // attention (batch, head, seq, seq)
    Tensor* K_h_tr = K_h->transpose(-1,-2);
    Tensor* attention_score = Q_h->matmul(K_h_tr);

    //divide dk sqrt
    float d_k_sqrt = static_cast<float>(sqrt(d_k));
    Tensor* attention_dk = attention_score->div(d_k_sqrt);
    //masking
    bool need_mask = static_cast<bool>(this->need_mask->data[0]);
    Tensor* attention_masked;
    if (need_mask == true) {
        attention_masked = attention_dk->add(this->causal_mask);
    } else attention_masked = attention_dk;
    //softmax
    Tensor* attention_softmax = this->softmax->forward(attention_masked);
    //attention_softmax matmul V_h (batch, head, seq, d_k)
    Tensor* V_adjusted = attention_softmax->matmul(V_h);
    // (batch, seq, head, d_k)
    Tensor* V_adjusted_tr = V_adjusted->transpose(-2,-3);
    Tensor* V_cont = V_adjusted_tr->contiguous();
    // (batch, seq, embed)
    Tensor* V_correct = V_cont->shape_view(input->shape);
    // (batch, seq, embed)
    Tensor* result = this->output_dense->forward(V_correct);
    return result;
}

void MultiheadAttentionBlock::init_members(){
    assert (this->parameters.size() == 3);
    this->heads = this->parameters[0];
    this->need_mask = this->parameters[1];
    this->causal_mask = this->parameters[2];
    this->causal_mask = this->parameters[2];
    assert (this->sub_blocks.size() == 5);
    this->query_dense = this->sub_blocks[0];
    this->key_dense = this->sub_blocks[1];
    this->value_dense = this->sub_blocks[2];
    this->output_dense = this->sub_blocks[3];
    this->softmax = this->sub_blocks[4];
}