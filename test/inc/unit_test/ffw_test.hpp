#ifndef FFW_TEST
#define FFW_TEST

#include "prebuilt_blocks.hpp"

class FFW_TEST{


bool run_tests(){
    bool all_success = true;
    check(this->mul_and_add_test(), all_success);
    return all_success;
}


// see if simple ffw can learn a * b + c = d
// dense layer and relu (maybe dropout)
// shape: ()
bool mul_and_add_test(){
    int sample_size = 20;
    std::vector<std::vector<Tensor*>> samples;
    for (int i = 0; i < sample_size; ++i){



    }    
    class FFW_Bloc




}



#endif