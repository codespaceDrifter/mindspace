//a simple dataset that directly uses a tensor in ram as the entire dataset. 
//used for testing. usually a dataset should be loaded from disk and then converted to tensors.
#ifndef TENSOR_DATASET_HPP
#define TENSOR_DATASET_HPP

#include "dataset.hpp"

class TensorDataset : public Dataset {

public:

Tensor* input;
Tensor* target;

TensorDataset (Tensor* input, Tensor* target){
    this->input = input;
    this->target = target;
}

std::size_t len () const override{
    std::vector<int> shape = this->input->shape;
    return static_cast<std::size_t>(shape[0]);
}

std::vector<Tensor*> getItem(std::size_t index) const override {
    std::vector<std::vector<int>> slices;
    slices.push_back({static_cast<int>(index), static_cast<int>(index + 1)});
    for (int i = 1; i < this->input->shape.size(); ++i) {
        slices.push_back({});
    }

    std::vector<Tensor*> result;
    Tensor* input_slice = this->input->slice(slices);
    Tensor* target_slice = this->target->slice(slices);
    result.push_back(input_slice);
    result.push_back(target_slice);
    return result;
}


};




#endif