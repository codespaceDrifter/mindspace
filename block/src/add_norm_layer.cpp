#include "add_norm_layer.hpp"

AddNormLayer::AddNormLayer (Block* component){
    this->type = "ADD_NORM_LAYER";
    this->register_block<AddNormLayer>();
    this->sub_blocks.push_back(component);
    this->init_members();
}

Tensor* AddNormLayer::forward_ (Tensor* input, Tensor* input2){
    input->set_anchor_true();
    Tensor* component_output = this->component->forward(input);

    Tensor* combine = input->add(component_output);

    float layer_dim = static_cast<float>(combine->shape[combine->shape.size()-1]);
    float episolon = 1e-5;
    Tensor* combine_reduce = combine->reduce_sum(-1);
    Tensor* mean = combine_reduce->div(layer_dim);
    Tensor* mean_diff = combine -> minus (mean);
    Tensor* mean_diff_sqr = mean_diff->pow(2);
    Tensor* mean_diff_sqr_reduce = mean_diff_sqr->reduce_sum(-1);
    Tensor* variance = mean_diff_sqr_reduce->div(layer_dim);
    Tensor* stdev = variance->pow(0.5);
    Tensor* stdev_epi = stdev->add(episolon);
    Tensor* z_score = mean_diff ->div(stdev_epi);

    input->set_anchor_false();


    return z_score;
}

void AddNormLayer::init_members(){
    assert (this->sub_blocks.size() == 1);
    this->component = this->sub_blocks[0];
}
