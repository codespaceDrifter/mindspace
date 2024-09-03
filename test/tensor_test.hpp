#ifndef TENSOR_TEST_HPP
#define TENSOR_TEST_HPP

#include "tensor.hpp"
#include "test_utils.hpp"
#include <iostream>
#include <string>

class TENSOR_TEST{
public:


bool run_tests(bool debug = false){
    bool all_success = true;

    check(this->create_test(debug), all_success);
    check(this->add_test(debug), all_success);
    check(this->matmul_test(debug), all_success);

    std::cout<<"tensor tests: " << bool_to_str(all_success) << std::endl;
    return all_success;
}


bool create_test(bool debug = false){
    bool success = true;
    Tensor *A = new Tensor();
    Tensor *B = new Tensor(1,2,3);
    check (all_equal(B->value_core.get(), 0), success);
    check (all_equal(B->grad_core.get(), 0), success);

    B->one_grad();
    check (all_equal(B->grad_core.get(),1), success);
    delete A;
    delete B;
    if (debug == true) std::cout<<"create tests: "<<bool_to_str(success)<<std::endl;
    return success;
}

bool add_test(bool debug = false){
    bool success = true;
    Tensor* A = new Tensor (3,2,4);
    Tensor* B = new Tensor (1,1,4);
    A->value_core->fill(1.1);
    B->value_core->fill(2.1);
    Tensor* C = A->add(B);
    check (all_equal(C->value_core.get(), 3.2), success);
    C->one_grad();
    C->backprop();
    check (all_equal(A->grad_core.get(), 1), success);
    check (all_equal(B->grad_core.get(), 1), success);
    delete A;
    delete B;
    delete C;

    if (debug == true) std::cout<<"add tests: "<<bool_to_str(success)<<std::endl;
    return success;
}

bool matmul_test (bool debug = false){
    bool success = true;

    Tensor* A = new Tensor (2,2,2);
    Tensor* B = new Tensor (2,3);
    A->value_core->arrange();
    B->value_core->arrange();
    Tensor* C = A->matmul(B);
    std::vector<float> correct_C_value {3,4,5,9,14,19,15,24,33,21,34,47};
    check (vec_all_equal(C->value_core.get(), correct_C_value), success);
    C->one_grad();
    C->backprop();
    std::vector<float> correct_A_grad {3,12,3,12,3,12,3,12};
    std::vector<float> correct_B_grad {12,12,12,16,16,16};
    check (vec_all_equal(A->grad_core.get(), correct_A_grad), success);
    check (vec_all_equal(B->grad_core.get(), correct_B_grad), success);

    Tensor* D = new Tensor (2,1);
    Tensor* E = new Tensor (2,1,2);
    D->value_core->arrange();
    E->value_core->arrange();
    Tensor* F = D->matmul(E);
    std::vector<float> correct_F_value {0,0,0,1,0,0,2,3};
    check (vec_all_equal(F->value_core.get(), correct_F_value), success);
    F->init_grad();
    F->grad_core->arrange();
    F->backprop();
    std::vector<float> correct_D_grad {24,36};
    std::vector<float> correct_E_grad {2,3,6,7};
    check (vec_all_equal(D->grad_core.get(), correct_D_grad), success);
    check (vec_all_equal(E->grad_core.get(), correct_E_grad), success);

    std::cout<<"value core D E F"<<std::endl;
    D->value_core->print();
    E->value_core->print();
    F->value_core->print();

    std::cout<<"grad core D E F"<<std::endl;
    D->grad_core->print();
    E->grad_core->print();
    F->grad_core->print();

    if (debug == true) std::cout<<"matmul tests: "<<bool_to_str(success)<<std::endl;
    return success;
}
};

#endif