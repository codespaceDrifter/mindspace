#include "tensor.hpp"
#include "chrono"

/*
in a hurry. will make this more organized (add a bench.hpp and cpp file) later
*/

void matmul_bench (int seconds = 10, int tensor_size = 1000){
    const int iterations = 1'000'000;

    Tensor* A = new Tensor (tensor_size, tensor_size);
    Tensor* B = new Tensor (tensor_size, tensor_size);
    A->randomize(-10,10);
    B->randomize(-10,10);

    auto start = std::chrono::high_resolution_clock::now();
    int i = 0;
    for (; i < iterations; ++i) {

        std::cout<<"iteration: "<<i<<std::endl;

        Tensor* output = A->matmul(B);
        delete output;
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= seconds) {
            break;
        }
    }
    std::cout<<"In "<<seconds <<" seconds. Tensors size "<<tensor_size<<" . Matmuls done: "<<i<<std::endl;

}

void add_latency_bench (int tensor_size = 1000){

    Tensor* A = new Tensor (tensor_size, tensor_size);
    Tensor* B = new Tensor (tensor_size, tensor_size);
    A->randomize(-10,10);
    B->randomize(-10,10);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor* output = A->add(B);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    float seconds_took = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    delete A;
    delete B;
    delete output;
 
    std::cout<<"add of size: "<<tensor_size<<" seconds took: "<<seconds_took<<std::endl;
}

void idx_latency_bench (int tensor_size = 1000){

    Tensor* A = new Tensor (tensor_size, tensor_size);
    Tensor* B = new Tensor (tensor_size, tensor_size);
    Tensor* C = new Tensor (tensor_size, tensor_size);
    A->randomize(-10,10);
    B->randomize(-10,10);
    C->randomize(-10,10);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < C->shape_size; ++i){
        std::vector<int> indices = C->f_s(i);
        float a = A->idx(A->f_s(i));
        //float b = B->idx(B->f_s(i));
        //float c = C->idx(C->f_s(i));
    }

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    float seconds_took = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    delete A;
    delete B;
    delete C;

    std::cout<<"(3 tensors 1 indice) idx of size : "<<tensor_size<<" seconds took: "<<seconds_took<<std::endl;
}

void float_array_add_latency (int tensor_size = 1000){

    Tensor* A_tensor = new Tensor (tensor_size, tensor_size);
    Tensor* B_tensor = new Tensor (10, tensor_size);
    Tensor* C_tensor = new Tensor (tensor_size, tensor_size);
    A_tensor->randomize(-10,10);
    B_tensor->randomize(-10,10);
    C_tensor->randomize(-10,10);

    float * A = A_tensor->data;
    float * B = B_tensor->data;
    float * C = C_tensor->data;

    auto start = std::chrono::high_resolution_clock::now();

    int i_B = 0;
    const int ten = B_tensor->shape[0];

    for (int i = 0; i < A_tensor->shape_size; ++i){
        if (++i_B == ten) i_B = 0;
        //C[i] = A[i] * B[i];
    }


    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    float seconds_took = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    delete A_tensor;
    delete B_tensor;
    delete C_tensor;
 
    std::cout<<"float array in tensors of sizes: "<<tensor_size<<" Seconds took: "<<seconds_took<<std::endl;


}