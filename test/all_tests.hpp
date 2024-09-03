#ifndef ALL_TESTS_HPP
#define ALL_TESTS_HPP

#include "tensor_test.hpp"
#include "tensor_core_test.hpp"

bool run_all_tests(){
    bool all_success;

    TENSOR_CORE_TEST tensor_core_test;
    TENSOR_TEST tensor_test;
    check(tensor_core_test.run_tests(), all_success);
    check( tensor_test.run_tests(), all_success);
    std::cout<<"ALL TESTS: " <<bool_to_str(all_success) << std::endl; 

    return all_success;
}
/*
windows command to count line numbers, not including test folder:

Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch "\\.vscode\\|\\cmake\\|\\test\\" } | Get-Content | Measure-Object -Line


*/

#endif