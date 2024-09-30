#include "block_tests.hpp"


void BlockTest::run_tests(){
    this->dense_layer_test();
    this->relu_test();
    this->dropout_test();
    this->ffw_test();
    this->save_load_test();
    this->embedding_test();

    std::cout<< "BLOCK TESTS: " << bool_to_str(all_passed);
}

void BlockTest::dense_layer_test(){
    bool success = true;

    Tensor* input = new Tensor (2,3);
    input->arrange();
    Block* dense = new DenseLayer(3,1);
    dense->parameters[0]->arrange();

    Tensor * result = dense->forward (input);

    check (all_equal (result->shape, {2,1}), success);
    check (all_equal(result, {5,14}), success);

    result->backward_model();
    delete input;
    dense->delete_model();

    this->check_all_passed(success);
    std::cout<<"Dense Layer Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::relu_test(){
    bool success = true;

    Tensor* input = new Tensor (2,2);
    input->arrange(-2);
    Block* relu = new ReluLayer();
    
    Tensor* result = relu->forward(input);

    check (all_equal (result->shape, {2,2}), success);
    check (all_equal (result, {0,0,0,1}), success);


    result->backward_model();
    delete input;
    relu->delete_model();

    this->check_all_passed(success);
    std::cout<<"Relu Layer Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::dropout_test(){
    bool success = true;

    Tensor* input = new Tensor (2,2);
    input->fill(1);
    Block* dropout = new DropoutLayer(1);

    Tensor* result = dropout->forward(input);

    check (all_equal (result, {0,0,0,0}), success);
    this->check_all_passed(success);

    result->backward_model();
    delete input;
    dropout->delete_model();

    this->check_all_passed(success);
    std::cout<<"Dropout Layer Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::ffw_test(){
    bool success = true;

    Tensor* input = new Tensor (2,2);
    input->arrange();

    Block* ffwblock = new FFWBlock(2,3,0);
    ffwblock->sub_blocks[0]->parameters[0]->arrange(-3);

    Tensor* result = ffwblock->forward(input);

    check (all_equal (result->shape, {2,3}), success);
    check (all_equal (result, {0,1,2,0,0,4}), success);

    result->backward_model();
    delete input;
    delete ffwblock;

    this->check_all_passed(success);
    std::cout<<"FFW Block Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::save_load_test(){
    bool success = true;

    Block* ffwblock = new FFWBlock(2,3,0);
    ffwblock->sub_blocks[0]->parameters[0]->arrange(-3);

    ffwblock->save_model();
    ffwblock->delete_model();

    Block* ffwblock2 = Block::load_model();

    Tensor* input = new Tensor (2,2);
    input->arrange();

    Tensor* result = ffwblock2->forward(input);

    check (all_equal (result->shape, {2,3}), success);
    check (all_equal (result, {0,1,2,0,0,4}), success);

    result->backward_model(false);

    delete input;
    ffwblock2->delete_model();

    this->check_all_passed(success);
    std::cout<<"Save Load Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::embedding_test(){
    bool success = true;

    Tensor* input = new Tensor (2,2);
    Block* embed = new EmbeddingLayer(3,2);

    input->arrange();
    input->data[3] = 2;
    embed->parameters[0]->arrange();

    Tensor* result = embed->forward(input);

    check (all_equal (result->shape, {2,2,2}), success);
    check (all_equal (result, {0,1,2,3,4,5,4,5}), success);

    delete input;
    delete result;
    embed->delete_model();

    this->check_all_passed(success);
    std::cout<<"Embedding Test: "<<bool_to_str(success)<<std::endl;
}