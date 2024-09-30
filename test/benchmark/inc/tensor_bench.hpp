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

void float_array_add_latency (int sqrt_size = 1000){

    int size = sqrt_size * sqrt_size;

    float * A = new float [size];
    float * B = new float [size];
    float * C = new float [size];

    for (int i = 0; i < size; ++i){
        A[i] = 1.1;
        B[i] = 2.2;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i){
        C[i] = A[i] * B[i];
    }

    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    float seconds_took = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    delete A;
    delete B;
    delete C;
 
    std::cout<<"float array of size: "<<size<<" seconds took: "<<seconds_took<<std::endl;


}