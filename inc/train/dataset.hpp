#ifndef DATASET_HPP
#define DATASET_HPP

#include "block.hpp"

class DataSet{
public:
virtual std::vector<Tensor*> get_item(int idx) = 0;

int len;
};

#endif