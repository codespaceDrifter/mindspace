#include "operations.hpp"
#include "tensor.hpp"
#include "all_tests.hpp"
#include "fstream"

int main (){


    for (int i = 0; i < 100; ++i){
        run_all_tests();
        std::cout<<"TEST RUN NUM: "<< i<<"  ";
    }

    //run_all_tests();

    return 0;
}