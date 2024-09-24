#include "tensor.hpp"
#include "operations.hpp"



namespace Op{


platforms get_platforms(){

    return platforms::Cpu;
}

void add (Tensor* A, Tensor*B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    element_create(A, B, output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            add_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}

void minus (Tensor* A, Tensor*B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    element_create(A, B, output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            minus_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}

void mul (Tensor* A, Tensor*B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    element_create(A, B, output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            mul_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}

void div (Tensor* A, Tensor*B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    assert (div_viable (B));
    element_create(A, B,output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            div_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}

void pow (Tensor*A, Tensor*B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    element_create(A, B, output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            pow_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}

void log (Tensor*A, Tensor*B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    assert (log_viable(A, B));
    element_create(A, B, output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            log_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}

void reduce_sum (Tensor* A, std::vector<int> target_shape, Tensor*& output){
    assert (target_shape.size() <= A->shape.size());
    int size_diff = A->shape.size() - target_shape.size();
    for (int i = 0; i < target_shape.size(); ++i){
        assert (target_shape[i] <= A->shape[i+ size_diff] && target_shape[i] >= 0);
    }

    assert (A != output);
    output = new Tensor (target_shape);

    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            reduce_sum_cpu(A, target_shape, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}

void compare (Tensor*A, Tensor*B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    element_create(A, B, output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            compare_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}


void max (Tensor* A, Tensor* B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    element_create(A, B, output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            max_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}


void min (Tensor* A, Tensor* B, Tensor*& output){
    assert (element_op_viable (A,B,output));
    element_create(A, B, output);
    platforms cur_platform = get_platforms();
    switch (cur_platform){
        case platforms::Cpu:
            min_cpu(A, B, output);
            break;
        default:
            throw std::runtime_error("Unsupported platform");
            break;
    }
}

//unoptimized operations

void matmul (Tensor* A, Tensor*B, Tensor*& output){

    assert (matmul_viable (A,B));
    assert (A != output && B != output);

    std::unique_ptr<Tensor> A_view = A->unsqueeze(-2);
    std::unique_ptr<Tensor> B_view = B->transpose()->unsqueeze(-3);


    Tensor* C;
    mul (A_view.get(), B_view.get(), C);

    Tensor* C_reduced;
    reduce_sum (C, -1, C_reduced);

    C_reduced->squeeze_(-1);
    delete C;
    output = C_reduced;
}

//operations other forms

void add (Tensor* A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); add (A, b, output); delete b;};
void minus (Tensor* A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); minus (A, b, output); delete b;};
void minus (float A, Tensor* B, Tensor*& output){Tensor* a = Tensor::make_a_num(A); minus (a, B, output); delete a;};
void mul (Tensor* A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); mul (A, b, output); delete b;};
void div (Tensor* A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); div (A, b, output); delete b;};
void pow (Tensor* A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); pow (A, b, output); delete b;};
void log (Tensor*A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); log (A, b, output); delete b;};
void compare (Tensor* A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); compare (A, b, output); delete b;};
void max (Tensor* A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); max (A, b, output); delete b;};
void min (Tensor* A, float B, Tensor*& output){Tensor* b = Tensor::make_a_num(B); min (A, b, output); delete b;};

void reduce_sum (Tensor* A, int target_dim, Tensor*& output){
    if (target_dim < 0) target_dim = A->shape.size() + target_dim;
    assert (A->shape.size() > target_dim && target_dim > 0);
    std::vector<int> result_shape = A->shape;
    result_shape[target_dim] = 1;
    reduce_sum(A, result_shape, output);
}


//create output tensor if not in place
void element_create (Tensor*A, Tensor*B, Tensor*& output){
    if (A == output || B == output){
        return;
    }
    std::vector<int> output_shape = A->max_broadcast_shape(B);
    output = new Tensor (output_shape);
}


//viability check

bool appro_equal (float a, float b, float episolon){
    return std::fabs (a - b) < episolon;
}

bool element_op_viable(Tensor*A, Tensor*B, Tensor* output){


    std::vector<int> A_padded = A->shape;
    std::vector<int> B_padded = B->shape;
    A_padded.insert(A_padded.begin(), std::max(0,static_cast<int>(B_padded.size() - A_padded.size())), 1);
    B_padded.insert(B_padded.begin(), std::max(0,static_cast<int>(A_padded.size() - B_padded.size())),1);
    for (int i = 0; i < A_padded.size(); ++i){
        if (A_padded[i] != B_padded[i] && A_padded[i] != 1 && B_padded[i] != 1){
            return false;
        }
    }

    //does not allow in place operations at the same time as lazy broadcasting
    if (A == output || B == output){
        if (A->shape.size() != B->shape.size()) return false;
        for (int i = 0; i < A->shape.size(); ++i){
            if (A->shape[i] != B->shape[i]) return false;
        }
    } 
    return true;    
}

bool div_viable(Tensor*B){
    for (int i = 0; i < B->shape_size; ++i){
        float cur_num = B->idx(B->f_s(i));
        if (appro_equal(cur_num,0) == true) return false;
    }
    return true;
}

bool log_viable (Tensor*A, Tensor*B){

    for (int i = 0; i < A->shape_size; ++i){
        float cur_num = A->idx(A->f_s(i));
        if (cur_num < 0) return false;
    }
    for (int i = 0; i < B->shape_size; ++i){
        float cur_num = B->idx(B->f_s(i));
        if (cur_num < 0) return false;
        if (appro_equal(cur_num,1) == true) return false;
    }
    return true;
}



bool matmul_viable (Tensor*A, Tensor*B){
    std::vector<int> A_padded = A->shape;
    std::vector<int> B_padded = B->shape;
    A_padded.insert(A_padded.begin(), std::max(0,static_cast<int>(B_padded.size() - A_padded.size())), 1);
    B_padded.insert(B_padded.begin(), std::max(0,static_cast<int>(A_padded.size() - B_padded.size())),1);
    for (int i = 0; i < A_padded.size() - 2; ++i){
        if (A_padded[i] != B_padded[i] && A_padded[i] != 1 && B_padded[i] != 1) return false;
    }
    if (A_padded[A_padded.size()-1] != B_padded[B_padded.size()-2]) return false;
    return true;
}
}