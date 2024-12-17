#include "tensor.hpp"
#include "operations.hpp"


Tensor::Tensor (std::vector<int> shape, bool requires_grad){
    this->owns_data = true;
    this->data = nullptr;
    this->shape = shape;
    this->tensor_init();
    this->viewed = nullptr;
    this->requires_grad = requires_grad;
    this->grad = nullptr;
    this->operation = Operation::None;
    this-> ID = -1;
    this->anchor = false;
}

Tensor::~Tensor (){

    if (this->owns_data == true){
        delete[] this->data;
    }

    for (Tensor* cur_operand : this->operands){
        if (cur_operand == nullptr) continue;
        for (int i = 0; i < cur_operand->outputs.size(); ++i){
            if (cur_operand->outputs[i] == this){
                cur_operand->outputs.erase(cur_operand->outputs.begin()+i);
                break;
            }
        }
    }

    for (Tensor* cur_output : this->outputs){
        if (cur_output == nullptr) continue;
        for (int i = 0; i < cur_output->operands.size(); ++i){
            if (cur_output->operands[i] == this){
                cur_output->operands.erase(cur_output->operands.begin()+i);
                break;
            }
        }
    }
    if (this->requires_grad == true){
        delete this->grad;
    }
}

void Tensor::data_init(int new_size){
    assert (this->owns_data == true);
    assert (new_size >= 0);
    this->data_size = new_size;
    if (this->data != nullptr){
        delete[] this->data;
    }
    float* new_data = new float[new_size];
    for (int i = 0; i < new_size; ++i){
        new_data[i] = 0;
    }
    this->data = new_data;
}

void Tensor::data_switch (float* new_data){
    assert (this->owns_data == true);
    delete[] this->data;
    this->data = new_data;
}

void Tensor::compute_stride (){
    this->stride.clear();
    this->lazy_stride.clear();
    int cur_stride = 1;
    for (int i = this->shape.size()-1 ; i >= 0; --i){
        this->stride.insert(stride.begin(), cur_stride);
        cur_stride *= this->shape[i];
    }
}

void Tensor::compute_lazy_stride (){
    this->lazy_stride.clear();
    for (int i = 0; i < this->shape.size(); ++i){
        if (this->shape[i] == 1){
            this->lazy_stride.push_back(0);
        } else {
            this->lazy_stride.push_back(this->stride[i]);
        }
    }
}

void Tensor::offset_init (){
    this->offset.clear();
    offset.insert(offset.begin(), this->shape.size(), 0);
}

void Tensor::compute_shape_size(){
    int shape_size = std::min(1, static_cast<int>(this->shape.size()));
    for (int i = 0; i < this->shape.size(); ++i){
        shape_size *= this->shape[i];
    }
    this->shape_size = shape_size;
}

void Tensor::tensor_init(){
    assert (this->owns_data == true);
    this->compute_shape_size();
    this->data_init(this->shape_size);
    this->data_size = this->shape_size;
    this->compute_stride();
    this->compute_lazy_stride();
    this->offset_init();
}


void Tensor::deep_equal (Tensor* other){
    Tensor* temp = new Tensor (other->shape, other->requires_grad);
    for (int i = 0; i < other->shape_size; ++i){
        std::vector<int> indices = temp->f_s(i);
        temp->idx(indices) = other->idx(indices);
    }

    this->shape = other->shape;
    this->tensor_init();
    for (int i = 0; i < temp->data_size; ++i){
        this->data[i] = temp->data[i];
    }
    delete temp;
}

Tensor* Tensor::make_a_num (float val){
    std::vector<int> result_shape {1};
    Tensor* result = new Tensor (result_shape);
    float* result_value = new float[1];
    result_value [0] = val;
    result->data_switch(result_value);
    return result;
}

void Tensor::randomize (float low, float high){
    for (int i = 0; i< this->data_size; ++i){
        float cur = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * (high - low) +low;
        this->data[i] = cur;
    }
}

void Tensor::fill (float value){
    for (int i = 0; i < this->data_size; ++i){
        this->data[i] = value;
    }
}

void Tensor::arrange (float start, float step){
    for (int i = 0; i < this->data_size; ++i){
        this->data[i] = start + i * step;
    }
}




std::vector<int> Tensor::max_broadcast_shape (Tensor* other) const{
    std::vector<int> this_padded = this->shape;
    std::vector<int> other_padded = other->shape;
    this_padded.insert(this_padded.begin(), std::max(0, static_cast<int>(other_padded.size() - this_padded.size()) ), 1);
    other_padded.insert(other_padded.begin(), std::max(0, static_cast<int>(this_padded.size() - other_padded.size())),1);
    std::vector<int> result;
    for (int i = 0; i < this_padded.size(); ++i){
        assert (this_padded[i] == 1 || other_padded[i] == 1 || this_padded[i] == other_padded[i]);
        result.push_back(std::max(this_padded[i], other_padded[i]));
    }
    return result;
}

Tensor* Tensor::vertical_stack(std::vector<Tensor*> tensor_vec) {
    // Verify shapes are compatible
    std::vector<int> shape_without_first(tensor_vec[0]->shape.begin()+1, tensor_vec[0]->shape.end());
    for (int i = 1; i < tensor_vec.size(); ++i) {
        assert(tensor_vec[i]->owns_data == true);
        std::vector<int> cur_shape_without_first(tensor_vec[i]->shape.begin()+1, tensor_vec[i]->shape.end());
        assert(cur_shape_without_first == shape_without_first);
    }

    // Calculate new shape
    std::vector<int> result_shape(shape_without_first);
    int total_first_dim = 0;
    for (int i = 0; i < tensor_vec.size(); ++i) {
        total_first_dim += tensor_vec[i]->shape[0];
    }
    result_shape.insert(result_shape.begin(), total_first_dim);
    
    // Create result tensor
    Tensor* result = new Tensor(result_shape, false);
    
    // Copy data taking strides into account
    int result_idx = 0;
    for (int i = 0; i < tensor_vec.size(); ++i) {
        Tensor* cur_tensor = tensor_vec[i];
        for (int j = 0; j < cur_tensor->shape_size; ++j) {
            std::vector<int> indices = cur_tensor->f_s(j);
            result->data[result_idx++] = cur_tensor->idx(indices);
        }
    }
    
    return result;
}

Tensor* Tensor::deep_broadcast(std::vector<int> target_shape){
    assert (target_shape.size() >= this->shape.size());
    int dim_diff = target_shape.size() - this->shape.size();
    for (int i = 0; i < this->shape.size(); ++i){
        assert (this->shape[i] == 1 || this->shape[i] == target_shape[i+dim_diff]);
    }

    Tensor* result = new Tensor (target_shape, this->requires_grad);
    for (int i = 0; i < result->shape_size; ++i){
        std::vector<int> indices = result->f_s(i);
        result->idx(indices) = this->idx(indices);
    }
    return result;
}

Tensor* Tensor::create_view(){
    Tensor* result = new Tensor({}, this->requires_grad);
    delete[] result->data;
    result -> data = this -> data;
    result -> data_size = this->data_size;
    result -> shape = this -> shape;
    result -> stride = this -> stride;
    result -> lazy_stride = this->lazy_stride;
    result -> offset = this -> offset;
    result -> compute_shape_size();
    result -> viewed = this;
    result -> owns_data = false;
    if (result->requires_grad == true){
        result-> operands.push_back(this);
        this-> outputs.push_back(result);
    }
    return result;
}

Tensor* Tensor::slice(std::vector<std::vector<int>> slices){
    assert (slices.size() == this->shape.size());
    Tensor* result = this->create_view();
    for (int i = 0; i < slices.size(); ++i){
        assert (slices[i].size() == 0 || slices[i].size() == 2);
        if (slices[i].size() == 2){
            int first = slices[i][0];
            int second = slices[i][1];
            assert (first >= 0 && second <= this->shape[i]);
            result->shape[i] = second - first;
            result->offset[i] += first;
        }
    }
    result->compute_shape_size();
    result->compute_lazy_stride();
    return result;
}

Tensor* Tensor::slice(std::initializer_list<std::initializer_list<int>> slices){
    std::vector<std::vector<int>> slices_vec (slices.size());
    int i = 0;
    for (std::initializer_list<int> inner_list : slices) {
        slices_vec[i] = std::vector<int>(inner_list);
        ++i;
    }
    return this->slice (slices_vec);
}


Tensor* Tensor::transpose(int dim1, int dim2){
    if (dim1 < 0) dim1 = this->shape.size() + dim1;
    if (dim2 < 0) dim2 = this->shape.size() + dim2;
    assert (dim1 >= 0 && dim1 < this->shape.size());
    assert (dim2 >= 0 && dim2 < this->shape.size());

    Tensor* result = this->create_view(); 
    std::swap(result->shape[dim1], result->shape[dim2]);
    std::swap (result->stride[dim1], result->stride[dim2]);
    std::swap (result->lazy_stride[dim1], result->lazy_stride[dim2]);
    return result; 
}

Tensor* Tensor::squeeze(int dim){
    if (dim < 0) dim = this->shape.size() + dim;
    assert (dim >= 0 && dim < this->shape.size());
    assert (this->shape[dim] == 1);
    Tensor* result = this->create_view();
    if (result->shape[dim] == 1){
        result->shape.erase(result->shape.begin() + dim);
        result->stride.erase(result->stride.begin() + dim);
        result->offset.erase(result->offset.begin() + dim);
    }
    result->compute_lazy_stride();
    return result;
}

Tensor* Tensor::unsqueeze(int dim){
    Tensor* result = this->create_view();
    if (dim < 0) dim = this->shape.size() + dim + 1;

    assert (dim >= 0 && dim <= this->shape.size());

    result->shape.insert(result->shape.begin() + dim, 1);

    int correct_stride = 1;
    for (int i = this->shape.size() - 1; i >= dim; --i){
        correct_stride *= this->shape[i];
    }
    result->stride.insert(result->stride.begin() + dim, correct_stride);
    result->offset.insert(result->offset.begin() + dim, 0);

    result->compute_lazy_stride();
    return result;
}

Tensor* Tensor::shape_view (std::vector<int> target_shape){
    assert (this->owns_data == true);
    Tensor* result = this->create_view();
    int new_data_size = 1;
    for (int i = 0; i < target_shape.size(); ++i){
        new_data_size *= target_shape[i];
    }
    assert(new_data_size == this->data_size);
    result->shape = target_shape;
    result->compute_shape_size();
    result->compute_stride();
    result->compute_lazy_stride();
    result->offset_init();
    return result;
}

Tensor* Tensor::contiguous(){
    Tensor* result = new Tensor (this->shape);
    //opeartion
    result->operation = Operation::Contigious;
    result->operands.push_back(this);
    this->outputs.push_back(result);

    //data correct ordering. the old shape of tensor need to be broadcastable into the new shape.
    for (int i = 0; i < this->shape_size; ++i){
        result->data[i] = this->idx(this->f_s(i));
    }
    return result;
}

void Tensor::squeeze_ (int dim){
    Tensor* viewed = this->squeeze(dim);
    this->shape = viewed->shape;
    this->stride = viewed->stride;
    this->lazy_stride = viewed->lazy_stride;
    this->offset = viewed->offset;
    delete viewed;
}

void Tensor::unsqueeze_ (int dim){
    Tensor* viewed = this->unsqueeze(dim);
    this->shape = viewed->shape;
    this->stride = viewed->stride;
    this->lazy_stride = viewed->lazy_stride;
    this->offset = viewed->offset;
    delete viewed;
}



void Tensor::init_grad (){
    assert (this->grad == nullptr);

    if (this->requires_grad == false) return;

    if (this->owns_data == true){
        this->grad = new Tensor (this->shape,false);
    } else {
        this->grad = this->viewed->grad->create_view();
        this->grad->shape = this->shape;
        this->grad->stride = this->stride;
        this->grad->lazy_stride = this->lazy_stride;
        this->grad->offset = this->offset;
        this->grad->shape_size = this->shape_size;
    }
}

void Tensor::topo_sort (std::set<Tensor*>& visited, std::vector<Tensor*>& sorted){
    if (visited.find(this) != visited.end()){
        return;
    }
    visited.insert(this);
    for (Tensor* operand : this->operands){
        if (operand == nullptr) continue;
        operand->topo_sort(visited, sorted);
    }
    sorted.push_back(this);
}

void Tensor::delete_intermediates(){
    std::set<Tensor*> visited;
    std::vector<Tensor*> sorted;
    this->topo_sort(visited, sorted);

    std::vector<Tensor*> reverse_sorted = sorted;
    std::reverse (reverse_sorted.begin(), reverse_sorted.end());

    for (int i = 1; i < reverse_sorted.size(); ++i){
        if (reverse_sorted[i] != nullptr && reverse_sorted[i]->operands.size() > 0 && reverse_sorted[i]->anchor == false){
            delete reverse_sorted[i];
            reverse_sorted[i] = nullptr;
        }
    }
}

void Tensor::backward_model (bool delete_intermediate){

    std::set<Tensor*> visited;
    std::vector<Tensor*> sorted;
    this->topo_sort(visited, sorted);

    std::vector<Tensor*> reverse_sorted = sorted;
    std::reverse (reverse_sorted.begin(), reverse_sorted.end());

    assert (this->owns_data == true && this->requires_grad == true);
    this->init_grad();
    this->grad->fill(1);

    for (int i = 0; i < sorted.size() - 1; ++i){
        sorted[i]->init_grad();
    }

    for (int i = 0; i < reverse_sorted.size(); ++i){
        reverse_sorted[i]->backprop();
    }

    if (delete_intermediate == true){
        this->delete_intermediates();
        delete reverse_sorted[0];
    }
}


void Tensor::graph_construction (Tensor* other, Tensor* result){
    result->operands.push_back(this);
    result->operands.push_back(other);
    this->outputs.push_back(result);
    other->outputs.push_back(result);
}


Tensor* Tensor::add (Tensor* other){
    Tensor* result = nullptr;
    Op::add(this, other, result);

    result->operation = Operation::Add;
    this->graph_construction(other, result);
    return result;
}

Tensor* Tensor::minus (Tensor* other){
    Tensor* result = nullptr;
    Op::minus(this, other, result);

    result->operation = Operation::Minus;
    this->graph_construction(other,result);
    return result;
}

Tensor* Tensor::mul (Tensor* other){
    Tensor* result = nullptr;
    Op::mul(this, other, result);

    result->operation = Operation::Mul;
    this->graph_construction(other,result);
    return result;
}

Tensor* Tensor::div (Tensor* other){
    Tensor* result = nullptr;
    Op::div(this, other, result);

    result->operation = Operation::Div;
    this->graph_construction(other,result);
    return result;
}

Tensor* Tensor::pow (Tensor* other){
    Tensor* result = nullptr;
    Op::pow(this, other, result);

    result->operation = Operation::Pow;
    this->graph_construction(other,result);
    return result;
}


Tensor* Tensor::reduce_sum (std::vector<int> target_shape){
    Tensor* result = nullptr;
    Op::reduce_sum(this, target_shape, result);

    result->operation = Operation::ReduceSum;
    result->operands.push_back(this);
    this->outputs.push_back(result);
    return result;
}

Tensor* Tensor::compare (Tensor* other){
    Tensor* result = nullptr;
    Op::compare(this, other, result);
    result->operation = Operation::None;
    this->graph_construction(other,result);
    return result;
}

Tensor* Tensor::max (Tensor* other){
    Tensor* result = nullptr;
    Op::max(this, other, result);

    result->operation = Operation::Max;
    this->graph_construction(other,result);
    return result;
}

Tensor* Tensor::min (Tensor* other){
    Tensor* result = nullptr;
    Op::min(this, other, result);

    result->operation = Operation::Min;
    this->graph_construction(other,result);
    return result;
}

Tensor* Tensor::reduce_sum(int target_dim){
    if (target_dim < 0) target_dim = this->shape.size() + target_dim;
    assert (this->shape.size() > target_dim && target_dim > 0);
    std::vector<int> result_shape = this->shape;
    result_shape[target_dim] = 1;
    return this->reduce_sum(result_shape);
}

Tensor* Tensor::matmul(Tensor* other){
    Tensor* result = nullptr;

    Op::matmul(this, other, result);

    result->operation = Operation::Matmul;
    result->operands.push_back(this);
    result->operands.push_back(other);
    return result;
}

void Tensor::add_ (Tensor* other) {Tensor* temp = this; Op::add(this, other, temp);};
void Tensor::minus_ (Tensor* other) {Tensor* temp = this; Op::minus (this, other, temp);};
void Tensor::mul_ (Tensor* other) {Tensor* temp = this;  Op::mul (this, other, temp);};
void Tensor::div_ (Tensor* other) {Tensor* temp = this; Op::div (this, other, temp);};


void Tensor::backprop (){
    if (this->requires_grad == false) return;

    switch (this->operation){
        case Operation::None:
            break;
        case Operation::Contigious:
            this->contigous_backprop();
            break;
        case Operation::Add:
            this->add_backprop();
            break;
        case Operation::Minus:
            this->minus_backprop();
            break;
        case Operation::Mul:
            this->mul_backprop();
            break;
        case Operation::Div:
            this->div_backprop();
            break;
        case Operation::Pow:
            this->pow_backprop();
            break;
        case Operation::ReduceSum:
            this->reduce_sum_backprop();
            break;
        case Operation::Max:
            this->max_backprop();
            break;
        case Operation::Min:
            this->min_backprop();
            break;
        case Operation::Matmul:
            this->matmul_backprop();
            break;
        default:
            throw std::runtime_error("Unsupported operation for backprop");
            break;
    }
}


void Tensor::accumulate_grad (Tensor* grad_A, Tensor* grad_B){

    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    if (operand_A != nullptr && operand_A->requires_grad == true){
        Tensor* reduced_grad_A = nullptr;
        Op::reduce_sum(grad_A, operand_A->shape, reduced_grad_A);
        Op::add(operand_A->grad, reduced_grad_A, operand_A->grad);
        delete reduced_grad_A;
    }

    if (operand_B != nullptr && operand_B->requires_grad == true){
        Tensor* reduced_grad_B = nullptr;
        Op::reduce_sum(grad_B, operand_B->shape, reduced_grad_B);
        Op::add(operand_B->grad, reduced_grad_B, operand_B->grad);
        delete reduced_grad_B;
    }
}

void Tensor::contigous_backprop(){
    if (this->requires_grad == true){
        this->operands[0]->add_(this->grad);
    }
}


void Tensor::add_backprop(){
    Tensor* grad_A = this->grad;
    Tensor* grad_B = this->grad;
    this->accumulate_grad(grad_A, grad_B);
}

void Tensor::minus_backprop(){
    Tensor* grad_A = this->grad;
    Tensor* grad_B = nullptr;
    if (this->operands[1]->requires_grad == true){
        Op::mul (this->grad, -1, grad_B);
    }
    this->accumulate_grad(grad_A, grad_B);
    delete grad_B;
}

void Tensor::mul_backprop(){
    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    Tensor* grad_A = nullptr;
    Tensor* grad_B = nullptr;
    if (operand_A->requires_grad == true){
        Op::mul(this->grad, operand_B, grad_A);
    }
    if (operand_B->requires_grad == true){
        Op::mul(this->grad, operand_A, grad_B);
    }

    this->accumulate_grad(grad_A, grad_B);
    delete grad_A;
    delete grad_B;
}

void Tensor::div_backprop(){
    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    Tensor* grad_A = nullptr;
    Tensor* grad_B = nullptr;

    if (operand_A -> requires_grad == true){
        Op::div (this->grad, operand_B, grad_A);
    }

    if (operand_B -> requires_grad == true){
        Tensor* b_squared = nullptr;
        Tensor* neg_a = nullptr;
        Tensor* neg_a_div_b_squared = nullptr;
        Op::pow (operand_B, 2, b_squared);
        Op::mul (operand_A,-1 , neg_a);
        Op::div (neg_a, b_squared, neg_a_div_b_squared);
        Op::mul (this->grad, neg_a_div_b_squared, grad_B);
        delete b_squared;
        delete neg_a;
        delete neg_a_div_b_squared;
    }

    this->accumulate_grad(grad_A, grad_B);
    delete grad_A;
    delete grad_B;
}

void Tensor::pow_backprop(){


    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    Tensor* grad_A = nullptr;
    Tensor* grad_B = nullptr;

    if (operand_A->requires_grad == true){
        Tensor* B_minus_one = nullptr;
        Tensor* A_pow_B_minus_one = nullptr;
        Op::minus (operand_B, 1, B_minus_one);
        Op::pow (operand_A, B_minus_one, A_pow_B_minus_one);
        Op::mul (operand_B, A_pow_B_minus_one, grad_A);
        delete B_minus_one;
        delete A_pow_B_minus_one;
    }

    if (operand_B->requires_grad == true){
        Tensor* A_pow_B = nullptr;
        Tensor* ln_A = nullptr;
        Op::pow(operand_A, operand_B, A_pow_B);
        float e = std::exp(1.0f);
        Op::log(operand_A, e, ln_A);
        Op::mul(A_pow_B, ln_A, grad_B);
        delete A_pow_B;
        delete ln_A;
    }

    this->accumulate_grad(grad_A, grad_B);
    delete grad_A;
    delete grad_B;
}

void Tensor::reduce_sum_backprop(){
    Tensor* operand_A = this->operands[0];
    if (operand_A->requires_grad == true){
        Tensor* broadcasted_this_grad = this->grad->deep_broadcast(operand_A->grad->shape);
        Op::add(operand_A->grad, broadcasted_this_grad, operand_A->grad);
        delete broadcasted_this_grad;
    }
}

void Tensor::max_backprop(){
    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    Tensor* A_comp_B = nullptr;
    Tensor* B_comp_A = nullptr;
    if (operand_A->requires_grad == true || operand_B->requires_grad == true){
        Op::compare(operand_A, operand_B, A_comp_B);
        Op::minus (1, A_comp_B, B_comp_A);
    }

    Tensor* grad_A = nullptr;
    Tensor* grad_B = nullptr;
    if (operand_A->requires_grad == true){
        Op::mul (this->grad, A_comp_B, grad_A);
    }
    if (operand_B->requires_grad == true){
        Op::mul (this->grad, B_comp_A, grad_B);
    }

    delete A_comp_B;
    delete B_comp_A;

    this->accumulate_grad(grad_A, grad_B);
    delete grad_A;
    delete grad_B;
}

void Tensor::min_backprop(){
    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    Tensor* A_comp_B_reversed = nullptr;
    Tensor* B_comp_A_reversed = nullptr;
    if (operand_A->requires_grad == true || operand_B->requires_grad == true){
        Op::compare(operand_B, operand_A, A_comp_B_reversed);
        Op::minus (1, A_comp_B_reversed, B_comp_A_reversed);
    }

    Tensor* grad_A = nullptr;
    Tensor* grad_B = nullptr;
    if (operand_A->requires_grad == true){
        Op::mul (this->grad, A_comp_B_reversed, grad_A);
    }
    if (operand_B->requires_grad == true){
        Op::mul (this->grad, B_comp_A_reversed, grad_B);
    }

    delete A_comp_B_reversed;
    delete B_comp_A_reversed;

    this->accumulate_grad(grad_A, grad_B);
    delete grad_A;
    delete grad_B;
}

void Tensor::matmul_backprop(){
    Tensor* operand_A = this->operands[0];
    Tensor* operand_B = this->operands[1];

    Tensor* grad_A = nullptr;
    Tensor* grad_B = nullptr;

    if (operand_A->requires_grad == true){
        Tensor* operand_B_T = operand_B->transpose();
        Op::matmul (this->grad, operand_B_T, grad_A);
        delete operand_B_T;
    }
    if (operand_B->requires_grad == true){
        Tensor* operand_A_T = operand_A->transpose();
        Op::matmul (operand_A_T, this->grad, grad_B);
        delete operand_A_T;
    }

    this->accumulate_grad(grad_A, grad_B);
    delete grad_A;
    delete grad_B;
}

std::vector<Tensor*> Tensor::anchored_tensors;
void Tensor::set_anchor_true(){
    if (this->anchor == true) return;
    this->anchor = true;
    Tensor::anchored_tensors.push_back(this);
    if (this->owns_data == false){
        this->viewed->set_anchor_true();
    }
}

void Tensor::set_anchor_false(){
    assert (this->anchor == true);
    for (int i = 0; i < anchored_tensors.size(); ++i){
        if (anchored_tensors[i] == this){
            anchored_tensors.erase(anchored_tensors.begin()+i);
            break;
        }
    }
    this->anchor = false;
}

void Tensor::clear_anchored(){
    for (Tensor* cur : Tensor::anchored_tensors){
        delete cur;
        cur = nullptr;
    }
    anchored_tensors.clear();
}


void Tensor::save (std::ofstream &out_file){
    assert (this->owns_data == true);
    if (!out_file) {
        std::cout << "Invalid ofstream object" << std::endl;
        return;
    }
    out_file.seekp(0,std::ios::end);
    out_file.write (reinterpret_cast<const char*> (&this->ID), sizeof(this->ID));
    int shape_size = this->shape.size();
    out_file.write (reinterpret_cast<const char*> (&shape_size), sizeof (shape_size));
    out_file.write (reinterpret_cast<const char*> (this->shape.data()), shape_size * sizeof (int));

    out_file.write (reinterpret_cast<const char*> (&this->data_size), sizeof (this->data_size));
    out_file.write (reinterpret_cast<const char*> (this->data), this->data_size * sizeof (float));
}

void Tensor::load (std::ifstream &in_file){
    assert (this-> data_size == 0);
    if (!in_file) {
        std::cout << "Invalid ifstream object" << std::endl;
        return;
    }
    in_file.read(reinterpret_cast<char*> (&this->ID), sizeof(this->ID));
    int shape_size;
    in_file.read (reinterpret_cast<char*> (&shape_size), sizeof (shape_size));
    this->shape.resize(shape_size);
    in_file.read (reinterpret_cast<char*> (this->shape.data()), shape_size * sizeof(int));
    this->tensor_init();
    in_file.read(reinterpret_cast<char*> (&this->data_size), sizeof(this->data_size));
    in_file.read (reinterpret_cast<char*> (this->data), this->data_size * sizeof (float));
}

void Tensor::print(){
    std::cout << "shape: (";
    for (int i = 0; i < this->shape.size(); ++i) {
        std::cout << this->shape[i];
        if (i != this->shape.size() - 1) std::cout << ", ";
    }
    std::cout <<")"<<std::endl;
    std::cout <<"data: "<<std::endl;
    std::cout <<this->to_string();
}

std::string Tensor::to_string(){
    if (this->shape.size() == 0 ) return "";

    std::vector<std::string> line_vec;
    std::string temp_str;

    for (int i = 0; i < this->shape_size; ++i){
        if (this->shape[this->shape.size()-1] == 1 || i != 0 && (i+1) %(this->shape[this->shape.size()-1]) == 0){
            temp_str += std::to_string(this->idx(this->f_s(i)));
            temp_str = "[" + temp_str + "]";
            line_vec.push_back(temp_str);
            temp_str = "";
        } else {
            temp_str = temp_str + std::to_string(this->idx(this->f_s(i))) + ", ";
        }
    }

    for (int i = this->shape.size() - 2; i > 0; --i){

        int cur_divide = this->shape[i];
        int cur_count = 0;
        int cur_max_divides = 1;
        int cur_divide_times = 1;

        for (int z = 0; z < i; ++z){
            cur_max_divides *= this->shape[z];
        }

        for (int j = 0; j < line_vec.size(); ++j) line_vec[j] = "  " + line_vec[j];

        for (int j = 0; j < line_vec.size(); ++j){
            if (i == this->shape.size() -2) cur_count ++;
            else if (line_vec[j] == "  ]") cur_count ++;
            if (cur_count / cur_divide == cur_divide_times && cur_count % cur_divide == 0 && cur_divide_times < cur_max_divides){
                ++ cur_divide_times;
                line_vec.insert(line_vec.begin()+j+1, "]");
                j++;
                line_vec.insert(line_vec.begin()+j+1, "[");
                j++;
            }
        }
        line_vec.insert(line_vec.begin(), "[");
        line_vec.push_back("]");
    }

    std::string result;
    for (int i = 0; i < line_vec.size(); ++i){
        result = result + line_vec[i] + "\n";
    }
    return result;
}

void Tensor::print_tree(int tabs){
    std::cout<<std::string (tabs, '\t');
    std::cout<<Tensor::vec_str(this->shape);
    std::cout<<": "<<Tensor::op_to_str(this->operation) << std::endl;
    tabs ++;
    for (Tensor* cur : this->operands){
        if (cur == nullptr) continue;
        cur->print_tree(tabs);
    }
}

std::string Tensor::vec_str(std::vector<int> vec){
    std::string result;
    result += "(";
    for (int i = 0; i < vec.size(); ++i){
        result += std::to_string(vec[i]);
        if (i != vec.size()-1) result += ", ";
    }
    result += ")";
    return result;
}

std::string Tensor::op_to_str (Operation op){
    switch (op) {
        case Operation::None:       return "None";
        case Operation::Add:        return "Add";
        case Operation::Minus:      return "Minus";
        case Operation::Mul:        return "Mul";
        case Operation::Div:        return "Div";
        case Operation::Pow:        return "Pow";
        case Operation::ReduceSum:  return "ReduceSum";
        case Operation::Max:        return "Max";
        case Operation::Min:        return "Min";
        case Operation::Matmul:     return "Matmul";
        default:                    return "Unknown";
    }
}