#include "tensor.hpp"

Tensor::Tensor (std::vector<int> shape){
    this->value_core.reset(TensorCore::createTensorCore(shape));
    this->grad_core.reset(TensorCore::createTensorCore(shape));
}

void Tensor::init_grad (){
    if (this -> grad_core->shape.size() == 0 && this->value_core->shape.size() > 0){
        this->grad_core->shape = this->value_core->shape;
        this->grad_core->shape_update();
        this->grad_core->fill(0);
    }
}

void Tensor::one_grad (){
    this->init_grad();
    this->grad_core->fill(1);
}

void Tensor::backprop (){
    for (Tensor* operand : this->grad_fn){
        operand->init_grad();
    }

    switch (this->op.type){
        case operationType::None:
            break;
        case operationType::Add:
            this->add_backprop();
            break;
        case operationType::Matmul:
            this->matmul_backprop();
            break;
        default:
            throw std::runtime_error("Unsupported operation for backprop");
            break;
    }
}

Tensor* Tensor::add (Tensor* other){

    Tensor* result = new Tensor();

    result->value_core.reset(this->value_core->add(other->value_core));
    result->grad_fn.push_back(this);

    result->grad_fn.push_back(other);
    result->op.type = operationType::Add;

    return result;
}

void Tensor::add_backprop(){
    for (Tensor* operand : this->grad_fn){
        operand->grad_core.reset(this->grad_core->add(operand->grad_core));
    }
}

Tensor* Tensor::matmul (Tensor* other){
    Tensor* result = new Tensor();
    result->value_core.reset(this->value_core->matmul(other->value_core));
    result->grad_fn.push_back(this);
    result->grad_fn.push_back(other);
    result->op.type = operationType::Matmul;
    return result;
}

void Tensor::matmul_backprop (){
    Tensor*& operand_A = this->grad_fn[0];
    Tensor*& operand_B = this->grad_fn[1];

    TensorCore* temp_grad_A = this->grad_core->matmul(operand_B->value_core->transpose().get());
    TensorCore* temp_grad_B = operand_A->value_core->transpose()->matmul(this->grad_core);

    TensorCore* grad_A = temp_grad_A->reduce_sum(operand_A->value_core->shape);
    TensorCore* grad_B = temp_grad_B->reduce_sum(operand_B->value_core->shape);

    delete temp_grad_A;
    delete temp_grad_B;

    operand_A ->grad_core.reset(operand_A->grad_core->add(grad_A));
    operand_B ->grad_core.reset(operand_B->grad_core->add(grad_B));
}

