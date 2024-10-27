#ifndef ALL_TESTS_HPP
#define ALL_TESTS_HPP

#include "tensor_tests.hpp"
#include "block_tests.hpp"

void run_all_tests(int run_times = 1){
    for (int i = 0; i < run_times; ++i){
        TensorTest tensor_test = TensorTest();
        tensor_test.run_tests();
        BlockTest block_test = BlockTest();
        block_test.run_tests();
    }
}


#endif



/*
windows command to count line numbers, not including test folder:

Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch "\\.vscode\\|\\cmake\\|\\test\\" } | Get-Content | Measure-Object -Line

try to keep under 1500 lines
*/