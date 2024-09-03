#ifndef TENSOR_CORE_TEST_HPP
#define TENSOR_CORE_TEST_HPP

#include "tensor_core.hpp"
#include "test_utils.hpp"
#include <iostream>
#include <string>


class TENSOR_CORE_TEST{
public:

bool run_tests(){
    bool all_success = true;
    check(this->create_test(), all_success);
    std::cout<<"tensor core tests: " << bool_to_str(all_success) << std::endl;
    return all_success;
}

bool create_test(){
    bool success = true;

    TensorCore* A = TensorCore::createTensorCore();
    check (A->data_size == 0, success);
    check (A->shape.size() == 0, success);

    std::vector<int> new_A_shape = {2,2,3}; 
    A->shape = new_A_shape;
    A->shape_update();
    check (A->data_size == 12, success);
    check (all_equal(A, 0), success);
    delete A;

    TensorCore* B = TensorCore::createTensorCore(5,1,1);
    check (B->data_size == 5, success);
    check (all_equal(B, 0), success);
    delete B;

    return success;
}
};


#endif