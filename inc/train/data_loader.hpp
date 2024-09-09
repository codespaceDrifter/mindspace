#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include "dataset.hpp"


//introduce CPU parallism in num works later
class DataLoader{
public:

DataLoader (DataSet* data_set, int batch_number, bool shuffle);

std::vector<Tensor*> get_data();

void shuffle_indexes();

DataSet* data_set;
int batch_number;
bool shuffle;
std::vector<int> indexes;
int cur_place;
};


#endif