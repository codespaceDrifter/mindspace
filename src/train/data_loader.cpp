#include "data_loader.hpp"
#include <random>

DataLoader::DataLoader (DataSet* data_set, int batch_number, bool shuffle){
    this->data_set = data_set;
    this->batch_number = batch_number;
    this->shuffle = shuffle;
    for (int i = 0; i < this->data_set->len; ++i){
        this->indexes.push_back(i);
    }
    cur_place = 0;
}


void DataLoader::shuffle_indexes(){
    std::random_device rd;
    std::mt19937 g (rd());
    std::shuffle(this->indexes.begin(), this->indexes.end(), g);
}

std::vector<Tensor*> DataLoader::get_data(){
    if (cur_place + batch_number >= this->indexes.size()) cur_place = 0;
    if (cur_place == 0 && this->shuffle == true) this->shuffle_indexes();

    std::vector<std::vector<Tensor*>> samples;
    for (; this->cur_place < batch_number; ++this->cur_place){
        samples.push_back(this->data_set->get_item(this->indexes[this->cur_place]));
    }

    std::vector<Tensor*> result;
    //i: 0 input,  i: 1 output
    for (int i = 0; i < samples[0].size(); ++i){
        std::vector<TensorCore*> cur_values;
        for (int j = 0; j < samples.size(); ++j){
            cur_values.push_back(samples[j][i]->value_core.get());
        }

        TensorCore* cur_value = TensorCore::vertical_stack(cur_values); 
        Tensor* cur_result = new Tensor (cur_value->shape);
        cur_result->value_core.reset(cur_value);
        result.push_back(cur_result);
    }

    for (int i = 0; i < samples.size(); ++i){
        for (int j = 0; j < samples[i].size(); ++j){
            delete samples[i][j];
        }
    }
    return result;
}

