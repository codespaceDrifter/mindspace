#include "tensor_tests.hpp"

void TensorTest::run_tests(){
    this->create_test();
    this->index_test();
    this->views_test();
    this->in_place_views_test();
    this->topo_sort_test();
    this->add_test();
    this->minus_test();
    this->mul_test();
    this->div_test();
    this->pow_test();
    this->reduce_sum_test();
    this->compare_test();
    this->max_test();
    this->min_test();
    this->matmul_test();
    this-> views_backprop_test();
    this->in_place_ops_test();
    this->save_load_test();
    std::cout<< "TENSOR TESTS: " << bool_to_str(all_passed);
}


void TensorTest::create_test(){
    bool success = true;

    Tensor* A = new Tensor ();
    check(A->shape.size() == 0, success);
    check(A->data_size == 0, success);
    delete A;
    if (this->debug == true) std::cout<<"Empty tensor made: " << bool_to_str(success);

    Tensor* B = new Tensor (2,2);
    check (B->data_size == 4, success);
    check (B->shape_size == 4, success);
    B->arrange ();
    check (all_equal(B, {0,1,2,3}), success);
    delete B;
    if (this->debug == true) std::cout<<"Tensor with shape made: " << bool_to_str(success);

    std::vector<int> C_shape = {3,1};
    float * C_data = new float[3];
    C_data[0] = 0; C_data[1] = 1; C_data[2] = 2;
    Tensor* C = new Tensor (C_shape);
    C->data_switch(C_data);
    check (all_equal(C, {0,1,2}), success);
    delete C;
    if (this->debug == true) std::cout<<"Tensor with data made: " << bool_to_str(success);

    this->check_all_passed(success);
    std::cout<<"Tensor Create Test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::index_test(){
   bool success = true;
   Tensor* A = new Tensor (2,3,1,4);
   A->arrange();
   for (float i = 0; i < A->shape_size; ++i){
        check (appro_equal(A->idx(A->f_s(i)), i), success);
    }

    this->check_all_passed(success);
    std::cout<<"Index Test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::views_test(){
    bool success = true;

    Tensor* A = new Tensor (2,2,2);
    A->arrange();

    std::unique_ptr<Tensor> B = A->slice({{0,1}, {}, {1,2}});
    check (all_equal(B->shape, {1,2,1}), success);
    check (all_equal(B.get(), {1,3}), success);
    if (this->debug == true) std::cout<<"slice: " << bool_to_str(success);

    std::unique_ptr<Tensor> C = A->transpose();
    check (all_equal(C->shape, {2,2,2}), success);
    check (all_equal(C.get(), {0,2,1,3,4,6,5,7}), success);
    if (this->debug == true) std::cout<<"transpose: " << bool_to_str(success);

    std::unique_ptr<Tensor> D = A->unsqueeze(1);
    check (all_equal(D->shape, {2,1,2,2}), success);
    check (all_equal(D.get(), A), success);
    if (this->debug == true) std::cout<<"unsqueeze: " << bool_to_str(success);

    std::unique_ptr<Tensor> E = D->squeeze(1);
    check (all_equal(E->shape, {2,2,2}), success);
    check (all_equal(E.get(), A), success);
    if (this->debug == true) std::cout<<"squeeze: " << bool_to_str(success);
    delete A;

    this->check_all_passed(success);
    std::cout<<"View Test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::in_place_views_test(){
    bool success = true;

    Tensor* A = new Tensor (2,2,2);
    A->arrange();

    A->slice_({{0,1}, {}, {1,2}});
    check (all_equal(A->shape, {1,2,1}), success);
    check (all_equal(A, {1,3}), success);
    if (this->debug == true) std::cout<<"slice: " << bool_to_str(success);

    A->transpose_();
    check (all_equal(A->shape, {1,1,2}), success);
    check (all_equal(A, {1,3}), success);

    A->unsqueeze_(-1);
    check (all_equal(A->shape, {1,1,2,1}), success);
    check (all_equal(A, {1,3}), success);

    A->squeeze_(0);
    check (all_equal(A->shape, {1,2,1}), success);
    check (all_equal(A, {1,3}), success);

    delete A;
    this->check_all_passed(success);
    std::cout<<"In Place View Test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::topo_sort_test(){
    bool success = true;
    Tensor* A = Tensor::make_a_num(1);
    Tensor* B = Tensor::make_a_num(10);
    Tensor* C = Tensor::make_a_num(100);

    Tensor* D;
    Tensor* E;
    Tensor* F;

    D = A->add(B);
    E = C->add(D);
    F = A->add(E);

    std::set<Tensor*> visited;
    std::vector<Tensor*> sorted;
    F->topo_sort(visited, sorted);

    check (sorted[0] == A, success);
    check (sorted[1] == C, success);
    check (sorted[2] == B, success);
    check (sorted[3] == D, success);
    check (sorted[4] == E, success);
    check (sorted[5] == F, success);

    this->check_all_passed(success);

    std::cout<<"Topo Sort Test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::add_test(){
    bool success = true;
    Tensor * A = new Tensor (2,1,2);
    Tensor * B = new Tensor (2,1);
    A->arrange();
    B->arrange();
    Tensor* C = A->add(B);
    check (all_equal(C->shape,{2,2,2}),success);
    check (all_equal(C, {0,1,1,2,2,3,3,4}), success);
    if (this->debug == true) std::cout<<"add: " <<bool_to_str(success);    

    C->backward_model();

    check (all_equal (A->grad, 2), success);
    check (all_equal (B->grad, 4), success);
    if (this->debug == true) std::cout<<"add backprop: " <<bool_to_str(success);    

    delete A; 
    delete B;

    this->check_all_passed(success);
    std::cout<<"Add Test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::minus_test(){
    bool success = true;
    Tensor * A = new Tensor (2);
    Tensor * B = new Tensor (2,1);
    A->arrange();
    B->arrange();

    Tensor* C = A->minus(B);
    check (all_equal(C->shape,{2,2}),success);
    check (all_equal(C, {0,1,-1,0}), success);
    C->backward_model();
    check (all_equal (A->grad, 2), success);
    check (all_equal (B->grad, -2), success);

    delete A; 
    delete B;

    this->check_all_passed(success);
    std::cout<<"minus test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::mul_test(){
    bool success = true;
    Tensor* A = new Tensor (2,2);
    Tensor* B = new Tensor (1,2);
    A->arrange();
    B->arrange();
    Tensor* C = A->mul(B);
    check (all_equal (C->shape, {2,2}),success);
    check (all_equal (C, {0,1,0,3}), success);

    C->backward_model();
    check (all_equal (A->grad, {0,1,0,1}), success);
    check (all_equal (B->grad, {2,4}), success);

    delete A;
    delete B;
    this->check_all_passed(success);
    std::cout<<"minus test: "<<bool_to_str(success)<<std::endl;
}


void TensorTest::div_test(){
    bool success = true;
    Tensor* A = new Tensor (2,2);
    Tensor* B = new Tensor (1,2);
    A->arrange();
    B->arrange(1);
    Tensor* C = A->div(B);

    check (all_equal (C->shape, {2,2}),success);
    check (all_equal (C, {0,0.5,2,1.5}), success);

    C->backward_model();
    check (all_equal (A->grad, {1,0.5,1,0.5}), success);
    check (all_equal (B->grad, {-2,-1}), success);

    delete A;
    delete B;
    this->check_all_passed(success);
    std::cout<<"div test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::pow_test(){
    bool success = true;
    Tensor* A = new Tensor (2,2);
    Tensor* B = new Tensor (1,2);
    A->arrange(1);
    B->arrange(1);
    Tensor* C = A->pow(B);

    check (all_equal (C->shape,{2,2}),success);
    check (all_equal(C, {1,4,3,16}), success);

    C->backward_model();
    check (all_equal(A->grad, {1,4,1,8}),success);
    check (all_equal(B->grad, {float (3 * std::log(3)), float(4 * std::log(2) + 16 * std::log(4) ) }) , success);

    delete A;
    delete B;

    this->check_all_passed(success);
    std::cout<<"pow test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::reduce_sum_test(){
    bool success = true;
    Tensor* A = new Tensor (2,2,2);
    A->arrange();

    std::vector<int> target_shape {1,2,1};
    Tensor* B = A->reduce_sum(target_shape);

    check (all_equal (B->shape, {1,2,1}), success);
    check (all_equal (B, {10,18}), success);

    B->backward_model();
    check (all_equal ( A->grad, 1), success);

    delete A;

    this->check_all_passed(success);
    std::cout<<"reduce sum test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::compare_test(){
    bool success = true;

    std::vector<int> shape = {2,2};
    float * a_data = new float[4];
    a_data[0] = 0; a_data[1] = 3; a_data[2] = 2; a_data[3] = -2;
    Tensor* A = new Tensor (shape);
    A->data_switch(a_data);
    float * b_data = new float[4];
    b_data[0] = 1; b_data[1] = -1; b_data[2] = 4; b_data[3] = -1;
    Tensor* B = new Tensor (shape);
    B->data_switch(b_data);
    Tensor* C = A->compare(B);
    check (all_equal (C, {0,1,0,0}), success);
    delete A;
    delete B;
    delete C;
    std::cout<<"compare test: "<<bool_to_str(success)<<std::endl;
}


void TensorTest::max_test(){
    bool success = true;
    std::vector<int> a_shape = {2,2};
    float* a_data = new float [4];
    a_data[0] = 0; a_data[1] = 3; a_data[2] = 2; a_data[3] = -2; 
    Tensor* A = new Tensor (a_shape);
    A->data_switch(a_data);

    std::vector<int> b_shape = {2,1};
    float* b_data = new float [2];
    b_data[0] = 1; b_data[1] = -1;
    Tensor* B = new Tensor (b_shape);
    B->data_switch(b_data);

    Tensor* C = A->max(B);
    check (all_equal (C, {1,3,2,-1}), success);
    delete A;
    delete B;
    delete C;

    this->check_all_passed(success);
    std::cout<<"max test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::min_test(){
    bool success = true;
    std::vector<int> a_shape = {2,2};
    float* a_data = new float [4];
    a_data[0] = 0; a_data[1] = 3; a_data[2] = 2; a_data[3] = -2; 
    Tensor* A = new Tensor (a_shape);
    A->data_switch(a_data);

    std::vector<int> b_shape = {2,1};
    float* b_data = new float [2];
    b_data[0] = 1; b_data[1] = -1;
    Tensor* B = new Tensor (b_shape);
    B->data_switch(b_data);

    Tensor* C = A->min(B);
    check (all_equal (C, {0,1,-1,-2}), success);

    delete A;
    delete B;
    delete C;

    this->check_all_passed(success);
    std::cout<<"min test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::matmul_test(){


    bool success = true;
    Tensor* A = new Tensor (2,2,2);
    Tensor* B = new Tensor (2,3);
    A->arrange();
    B->arrange();
    Tensor* C = A->matmul(B);
    std::vector<float> correct_C_value {3,4,5,9,14,19,15,24,33,21,34,47};
    check (all_equal(C, correct_C_value), success);
    
    C->backward_model();

    std::vector<float> correct_A_grad {3,12,3,12,3,12,3,12};
    std::vector<float> correct_B_grad {12,12,12,16,16,16};
    check (all_equal(A->grad, correct_A_grad), success);
    check (all_equal(B->grad, correct_B_grad), success);
    

    this->check_all_passed(success);
    std::cout<<"matmul tests: "<<bool_to_str(success)<<std::endl;
}


void TensorTest::views_backprop_test(){
    bool success = true;
    Tensor * A = new Tensor (2,2);
    A->arrange();
    Tensor* B = A->slice_r({{0,1},{}});
    Tensor* C = A->transpose_r();

    Tensor* D = B->add(C);

    check (all_equal(D, {0,3,1,4}), success);

    D->backward_model();

    check (all_equal(A->grad, {3,3,1,1}), success);

    delete A;

    this->check_all_passed(success);
    std::cout<<"views backprop test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::in_place_ops_test(){
    bool success = true;
    Tensor * A = new Tensor (2,2,2);
    A->arrange();

    Tensor * B = new Tensor (2,2,2);
    B->arrange();

    A->add_(B);
    check (all_equal (A, {0,2,4,6,8,10,12,14}), success);

    A->minus_(B);
    check (all_equal (A, {0,1,2,3,4,5,6,7}), success);

    A->mul_(B);
    check (all_equal (A, {0,1,4,9,16,25,36,49}), success);

    B->fill(2);
    A->div_(B);
    check (all_equal (A, {0, 0.5, 2, 4.5, 8, 12.5, 18, 24.5}), success);

    delete A;
    delete B;

    this->check_all_passed(success);
    std::cout<<"in place operations test: "<<bool_to_str(success)<<std::endl;
}

void TensorTest::save_load_test(){
    bool success = true;

    std::ofstream out_file("bin/save_load_test.bin", std::ios::binary);
    if (!out_file) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    Tensor * A = new Tensor (2,3);
    A->arrange();

    Tensor * B = new Tensor (1,2);
    B->arrange ();

    A->save(out_file);
    B->save(out_file);
    out_file.close();

    std::ifstream in_file ("bin/save_load_test.bin", std::ios::binary);
    if (!in_file) {
        std::cerr << "Error opening file for reading." << std::endl;
        return;
    }
    
    Tensor * C = new Tensor ();
    Tensor * D = new Tensor ();

    C->load (in_file);
    D->load (in_file);

    in_file.close();

    check (all_equal (A, C), success);
    check (all_equal (B,D), success);

    std::remove("bin/save_load_test.bin");

    delete A;
    delete B;
    delete C;
    delete D;


    this->check_all_passed(success);
    std::cout<<"save load test: "<<bool_to_str(success)<<std::endl;
}