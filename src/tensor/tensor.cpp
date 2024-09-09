#include "tensor.hpp"

Tensor::Tensor (std::vector<int> shape, bool intermediate){
    this->value_core.reset(TensorCore::create_TensorCore(shape));
    this->grad_core.reset(TensorCore::create_TensorCore(shape));
    this->intermediate = intermediate;
}

void Tensor::init_grad (){
    if (this -> grad_core->shape.size() == 0 && this->value_core->shape.size() > 0){
        this->grad_core->shape = this->value_core->shape;
        this->grad_core->contiguous();
        this->grad_core->fill(0);
    }
}

void Tensor::one_grad (){
    this->init_grad();
    this->grad_core->fill(1);
}

void Tensor::set_intermediate(){
    for (Tensor* operand: this->operands){
        if (operand->intermediate == true) this->intermediate = true;
    }
}

void Tensor::backprop (){
    for (Tensor* operand : this->operands){
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

    result->operands.push_back(this);
    result->operands.push_back(other);
    result->op.type = operationType::Add;
    result->set_intermediate();

    return result;
}

void Tensor::add_backprop(){
    // maybe do not use ptr references just use ptrs

    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    TensorCore* grad_A = this->grad_core->reduce_sum(operand_A->value_core->shape);
    TensorCore* grad_B = this->grad_core->reduce_sum(operand_B->value_core->shape);

    operand_A ->grad_core.reset(grad_A);
    operand_B ->grad_core.reset(grad_B);
}

Tensor* Tensor::mul (Tensor* other){
    Tensor* result = new Tensor();

    result->value_core.reset(this->value_core->mul(other->value_core));

    result->operands.push_back(this);
    result->operands.push_back(other);
    result->op.type = operationType::Mul;
    result->set_intermediate();
    return result;

}

void Tensor::mul_backprop(){
    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    TensorCore* temp_grad_A = this->grad_core->mul(operand_B->value_core);
    TensorCore* temp_grad_B = this->grad_core->mul(operand_A->value_core);
    
    TensorCore* grad_A = temp_grad_A->reduce_sum(operand_A->value_core->shape);
    TensorCore* grad_B = temp_grad_B->reduce_sum(operand_B->value_core->shape);

    delete temp_grad_A;
    delete temp_grad_B;

    operand_A ->grad_core.reset(operand_A->grad_core->add(grad_A));
    operand_B ->grad_core.reset(operand_B->grad_core->add(grad_B));
}

Tensor* Tensor::matmul (Tensor* other){
    Tensor* result = new Tensor();
    result->value_core.reset(this->value_core->matmul(other->value_core));
    result->operands.push_back(this);
    result->operands.push_back(other);
    result->op.type = operationType::Matmul;
    result->set_intermediate();
    return result;
}

void Tensor::matmul_backprop (){
    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    TensorCore* temp_grad_A = this->grad_core->matmul(operand_B->value_core->transpose().get());
    TensorCore* temp_grad_B = operand_A->value_core->transpose()->matmul(this->grad_core);

    TensorCore* grad_A = temp_grad_A->reduce_sum(operand_A->value_core->shape);
    TensorCore* grad_B = temp_grad_B->reduce_sum(operand_B->value_core->shape);

    delete temp_grad_A;
    delete temp_grad_B;

    operand_A ->grad_core.reset(operand_A->grad_core->add(grad_A));
    operand_B ->grad_core.reset(operand_B->grad_core->add(grad_B));
}

Tensor* Tensor::max (Tensor* other){
    Tensor* result = new Tensor();
    TensorCore* max_result = this->value_core->max(other->value_core);
    assert (this->value_core->shape == max_result->shape);
    result->value_core.reset(max_result);
    result->operands.push_back(this);
    result->operands.push_back(other);
    result->op.type = operationType::Max;
    result->set_intermediate();
    return result;
}

void Tensor::max_backprop (){
    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];
    TensorCore* A_comp_B = operand_A->value_core->compare(operand_B->value_core);
    TensorCore* B_comp_A = A_comp_B->toggle_bits();

    TensorCore* grad_A = A_comp_B ->mul (this->grad_core);
    TensorCore* grad_B = B_comp_A ->mul (this->grad_core);
    delete A_comp_B;
    delete B_comp_A;

    operand_A ->grad_core.reset(operand_A->grad_core->add(grad_A));
    operand_B ->grad_core.reset(operand_B->grad_core->add(grad_B));
}