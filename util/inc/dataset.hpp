#ifndef DATASET_HPP
#define DATASET_HPP


#include "tensor.hpp"

class Dataset {
public:
    virtual ~Dataset() = default;
    
    // Get the total number of items in the dataset
    virtual std::size_t len() const = 0;
    
    // Get item at specified index, returns pair of (input, target) tensors
    virtual std::vector<Tensor*> getItem(std::size_t index) const = 0;
};

#endif // DATASET_HPP

