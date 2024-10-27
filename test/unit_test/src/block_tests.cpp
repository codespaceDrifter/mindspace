#include "block_tests.hpp"


void BlockTest::run_tests(){
    this->dense_layer_test();
    this->relu_test();
    this->dropout_test();
    this->ffw_test();
    this->inference_test();
    this->save_load_test();
    this->embedding_test();
    this->pos_embed_test();
    this->add_norm_test();
    this->softmax_test();
    this->multiheaded_attention_test();
    this->encoder_block_test();
    this->decoder_block_test();
    this->transformer_model_test();
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

    Block* ffwblock = new FFWBlock(2,2,2,0);
    ffwblock->sub_blocks[0]->parameters[0]->arrange(0);
    ffwblock->sub_blocks[3]->parameters[0]->arrange(0);

    Tensor* result = ffwblock->forward(input);

    check (all_equal (result->shape, {2,2}), success);
    check (all_equal (result, {6,11,22,39}), success);

    result->backward_model();
    delete input;
    delete ffwblock;

    this->check_all_passed(success);
    std::cout<<"FFW Block Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::inference_test(){
    bool success = true;

    Tensor* input = new Tensor (2,2);
    input->arrange();

    Block* ffwblock = new FFWBlock(2,2,2,0);
    ffwblock->sub_blocks[0]->parameters[0]->arrange(0);
    ffwblock->sub_blocks[3]->parameters[0]->arrange(0);

    Block::training = false;

    Tensor* result = ffwblock->forward(input);

    check (all_equal (result->shape, {2,2}), success);
    check (all_equal (result, {6,11,22,39}), success);
    delete ffwblock;

    this->check_all_passed(success);
    Block::training = true;
    std::cout<<"Inference Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::save_load_test(){
    bool success = true;

    Block* ffwblock = new FFWBlock(2,2,2,0);
    ffwblock->sub_blocks[0]->parameters[0]->arrange(0);
    ffwblock->sub_blocks[3]->parameters[0]->arrange(0);

    ffwblock->save_model();
    ffwblock->delete_model();

    Block* ffwblock2 = Block::load_model();

    Tensor* input = new Tensor (2,2);
    input->arrange();

    Tensor* result = ffwblock2->forward(input);

    check (all_equal (result->shape, {2,2}), success);
    check (all_equal (result, {6,11,22,39}), success);

    result->backward_model();
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

    result->backward_model();
    delete input;
    embed->delete_model();

    this->check_all_passed(success);
    std::cout<<"Embedding Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::pos_embed_test(){
    bool success = true;

    Tensor* input = new Tensor (3,4);
    Block* pos_embed = new PosEmbedLayer(3,4);
    Tensor* result = pos_embed->forward(input);

    check (all_equal (result->shape, {3,4}), success);
    check (all_equal (result, {0,1,0,1,0.841471, 0.540302, 0.099833, 0.995004,0.909297, -0.416147, 0.198669, 0.980067 }),success);

    result->backward_model();
    delete input;
    pos_embed->delete_model();

    this->check_all_passed(success);
    std::cout<<"Pos Embed Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::add_norm_test(){
    bool success = true;

    Tensor* input = new Tensor (2,2);
    Block* dense = new DenseLayer(2,2);
    input->arrange();
    dense->parameters[0]->arrange();
    Block* add_norm = new AddNormLayer(dense);
    Tensor* result = add_norm ->forward(input);

    check (all_equal (result->shape, {2,2}), success);
    check (all_equal (result, {-1,1,-1,1}), success);
    
    result->backward_model();
    delete input;
    add_norm->delete_model();

    this->check_all_passed(success);
    std::cout<<"Add Norm Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::softmax_test(){
    bool success = true;

    Tensor* input = new Tensor (2,3);
    input->arrange();
    Block* softmax = new SoftmaxLayer();
    Tensor* result = softmax->forward(input);

    check (all_equal (result->shape, {2,3}), success);
    check (all_equal (result, {0.09, 0.245, 0.665, 0.09, 0.245, 0.665}), success);

    this->check_all_passed(success);
    std::cout<<"Softmax Test: "<<bool_to_str(success)<<std::endl;
}

void BlockTest::multiheaded_attention_test(){
    bool success = true;
    Tensor* input = new Tensor(2,2);
    input->arrange(0.1, 0.1);

    Block* mha = new MultiheadAttentionBlock(2,2,2,true);

    // Arrange dense layer parameters
    for (int i = 0; i < 4; ++i) {  
        mha->sub_blocks[i]->parameters[0]->arrange(0.1, 0.1);  
    }

    // Forward pass
    Tensor* result = mha->forward(input);

    // Check output shape
    check(all_equal(result->shape, {2, 2}), success);

    check(all_equal(result, {0.037,0.054,0.0592652,0.0863692}), success);

    // Backward pass
    result->backward_model();
    delete input;
    mha->delete_model();
    this->check_all_passed(success);
    std::cout<<"Multiheaded Attention Test: "<<bool_to_str(success)<<std::endl;
}


void BlockTest::encoder_block_test(){
    bool success = true;

    // Create input tensor
    Tensor* input = new Tensor(4, 8);
    input->arrange(0.1, 0.1);

    // Create EncoderBlock
    EncoderBlock* encoder_block = new EncoderBlock(4, 8, 2, 2, 0.1);

    // Forward pass
    Tensor* output = encoder_block->forward(input);

    // Check output shape
    check(all_equal(output->shape, {4, 8}), success);

    // Backward pass
    output->backward_model();
    // Clean up
    delete input;
    encoder_block->delete_model();

    this->check_all_passed(success);
    std::cout << "Encoder Block Test: " << bool_to_str(success) << std::endl;
}

void BlockTest::decoder_block_test(){
    bool success = true;
    // Create input tensors
    Tensor* input = new Tensor(4, 8);
    input->arrange(0.1, 0.1);
    Tensor* encoder_output = new Tensor(4, 8);
    encoder_output->arrange(0.2, 0.1);

    // Create DecoderBlock
    DecoderBlock* decoder_block = new DecoderBlock(4, 8, 2, 16, 0.1);

    // Forward pass
    Tensor* output = decoder_block->forward(input, encoder_output);

    // Check output shape
    check(all_equal(output->shape, {4, 8}), success);

    // Backward pass
    output->backward_model();

    // Clean up
    delete input;
    delete encoder_output;
    decoder_block->delete_model();

    this->check_all_passed(success);
    std::cout << "Decoder Block Test: " << bool_to_str(success) << std::endl;



}
void BlockTest::transformer_model_test() {
    bool success = true;

    // Create input tensors

    // Create TransformerModel
    int vocab_size = 10;
    int seq_len = 8;
    int embed_dim = 16;
    int num_heads = 2;
    int ffw_hidden = 32;
    int num_encoder_blocks = 2;
    int num_decoder_blocks = 2;
    float dropout_rate = 0.1;

    Tensor* encoder_input = new Tensor(seq_len);
    Tensor* decoder_input = new Tensor(seq_len);

    TransformerModel* transformer = new TransformerModel(vocab_size, seq_len, embed_dim, num_heads, ffw_hidden, 
                                                         num_encoder_blocks, num_decoder_blocks, dropout_rate);


    // Forward pass
    Tensor* output = transformer->forward(encoder_input, decoder_input);

    // Check output shape
    check(all_equal(output->shape, {seq_len, vocab_size}), success);

    // Backward pass
    output->backward_model();

    // Clean up
    delete encoder_input;
    delete decoder_input;
    transformer->delete_model();

    this->check_all_passed(success);
    std::cout << "Transformer Model Test: " << bool_to_str(success) << std::endl;
}