#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <vector>
#include <random>
#include "dataset.hpp"

class DataLoader {
public:
    Dataset* dataset;
    size_t batchSize;
    bool shuffle;
    std::vector<size_t> indices;
    std::mt19937 rng;
    size_t index;

    DataLoader(Dataset* dataset, size_t batchSize, bool shuffle) 
        : dataset(dataset), batchSize(batchSize), shuffle(shuffle), index(0) {
        indices.resize(dataset->len());
        for(size_t i = 0; i < dataset->len(); i++) {
            indices[i] = i;
        }
        rng = std::mt19937(std::random_device()());
    }

    std::vector<Tensor*> getNextBatch() {

        if(index + batchSize > dataset->len()) {
            index = 0;
        }
        
        if(index == 0 && shuffle) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        
        // sample outlerloop input and target innerloop
        std::vector<std::vector<Tensor*>> seperate_samples;
        for(size_t i = 0; i < batchSize; i++) {
            std::vector<Tensor*> cur_sample = dataset->getItem(indices[index + i]);
            seperate_samples.push_back(cur_sample);
        }

        //input target outerloop sample innerloop
        int components = seperate_samples[0].size();
        std::vector<std::vector<Tensor*>> seperate_components;
        for (int i = 0; i < components; i++) {
            std::vector<Tensor*> cur_component;
            for (int j = 0; j < seperate_samples.size(); j++){
                cur_component.push_back(seperate_samples[j][i]);
            }

            seperate_components.push_back(cur_component);
        }

        std::vector<Tensor*> result;
        for (int i = 0; i < components; ++i){
            Tensor* cur_component = Tensor::vertical_stack(seperate_components[i]);

            //maybe delete the pointers in each strip here?
            result.push_back(cur_component);
        }
        
        index += batchSize;
        return result;
    }
};

#endif // DATALOADER_HPP
