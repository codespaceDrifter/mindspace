//single core cpu with c++. no parallism

#include "tensor.hpp"
#include "operations.hpp"


namespace Op{

void add_cpu(Tensor* A, Tensor*B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        output->idx(indices) = A->idx(indices) + B->idx(indices);
    }
}

void minus_cpu(Tensor* A, Tensor* B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        output->idx(indices) = A->idx(indices) - B->idx(indices);
    }
}

void mul_cpu (Tensor* A, Tensor* B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        output->idx(indices) = A->idx(indices) * B->idx(indices);
    }
}

void div_cpu (Tensor* A, Tensor* B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        output->idx(indices) = A->idx(indices) / B->idx(indices);
    }
}



void pow_cpu (Tensor* A, Tensor* B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        output->idx(indices) = std::pow (A->idx(indices) , B->idx(indices));
    }
}

void log_cpu (Tensor* A, Tensor* B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        output->idx(indices) = std::log(A->idx(indices)) / std::log(B->idx(indices));
    }
}

void reduce_sum_cpu (Tensor* A, std::vector<int> target_shape, Tensor* output){
    for (int i = 0; i < A->shape_size; ++i){
        std::vector<int> indices = A->f_s(i);
        output->idx(indices) += A->idx(indices);
    }
}

void compare_cpu (Tensor* A, Tensor* B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        float A_num = A->idx(indices);
        float B_num = B->idx(indices);
        if (A_num >= B_num){
            output->idx(indices) = 1.0f;
        } else {
            output->idx(indices) = 0.0f;
        }
    }
}

void max_cpu (Tensor* A, Tensor* B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        float A_num = A->idx(indices);
        float B_num = B->idx(indices);
        if (A_num >= B_num){
            output->idx(indices) = A_num;
        } else {
            output->idx(indices) = B_num;
        }
    }
}

void min_cpu (Tensor* A, Tensor* B, Tensor* output){
    for (int i = 0; i < output->shape_size; ++i){
        std::vector<int> indices = output->f_s(i);
        float A_num = A->idx(indices);
        float B_num = B->idx(indices);
        if (A_num <= B_num){
            output->idx(indices) = A_num;
        } else {
            output->idx(indices) = B_num;
        }
    }

}


//write later
void matmul_cpu (Tensor* A, Tensor* B, Tensor* output){


}

}
