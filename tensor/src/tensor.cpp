#include "tensor.hpp"
#include "operations.hpp"


Tensor::Tensor (std::vector<int> shape, bool requires_grad){
    this->shape = shape;
    this->compute_data_size();
    this->data = nullptr;
    this->data_init();
    this->compute_stride();
    this->requires_grad = requires_grad;
    this->grad = nullptr;
    this->operation = Operation::None;
    this-> ID = -1;
}


Tensor::~Tensor (){
    delete[] this->data;
    delete this->grad;
}

void Tensor::compute_data_size(){
    int shape_size = std::min(1, static_cast<int>(this->shape.size()));
    for (int i = 0; i < this->shape.size(); ++i){
        shape_size *= this->shape[i];
    }
    this->data_size = shape_size;
}

void Tensor::data_init(){
    if (this->data != nullptr){
        delete[] this->data;
    }
    float* new_data = new float[this->data_size];
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

void Tensor::compute_stride(){
    this->stride.clear();
    int cur_stride = 1;
    for (int i = this->shape.size()-1 ; i >= 0; --i){
        this->stride.insert(stride.begin(), cur_stride);
        cur_stride *= this->shape[i];
    }
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
    std::fill (this->data, this->data + this->data_size, value);
}

void Tensor::arrange (int start){
    for (int i = 0; i < this->data_size; ++i){
        this->data[i] = start + i;
    }
}

void Tensor::print(){
    std::cout << "shape: (";
    for (int i = 0; i < this->shape.size(); ++i) {
        std::cout << this->shape[i];
        if (i != this->shape.size() - 1) std::cout << ", ";
    }
    std::cout <<")"<<std::endl;
    std::cout <<"data: "<<std::endl;
    std::cout<<this->to_string();
}

std::string Tensor::to_string(){
    if (this->shape.size() == 0 ) return "";

    std::vector<std::string> line_vec;
    std::string temp_str;

    int last_dim_tracker = 0
    for (int i = 0; i < this->data_size; ++i){
        last_dim_tracker ++;
        if (last_dim_tracker == this->shape[this->shape.size()-1]){
            temp_str += std::to_string(this->data[i]);
            temp_str = "[" + temp_str + "]";
            line_vec.push_back(temp_str);
            temp_str = "";
        } else {
            temp_str = temp_str + std::to_string(this->data[i] + ", ");
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

void Tensor::transpose (int dim1, int dim2){
    if (dim1 < 0) dim1 = this->shape.size() + dim1;
    if (dim2 < 0) dim2 = this->shape.size() + dim2;
    assert (dim1 >= 0 && dim1 < this->shape.size());
    assert (dim2 >= 0 && dim2 < this->shape.size());

    bool* transposed = new bool [this->data_size];
    std::fill(transposed, transposed + this->data_size, false);

    std::vector<int> cur_indices;
    int i_transposed;
    for (int i = 0; i < this->data_size; ++i){
        if (transposed[i] == false){
            cur_indices = this->f_s(i);
            std::swap (cur_indices[dim1], cur_indices[dim2]);
            i_transposed = 0;
            for (int j = 0; j < cur_indices.size(); ++j){
                i_transposed += cur_indices[j] * this->stride[j];
            }
            std::swap (this->data[i], this->data[i_transposed]);
            transposed[i] == true;
            transposed[i_transposed] == true;
        }
    }
    delete transposed;
    std::swap(this->shape[dim1], this->shape[dim2]);
    this->calculate_stride();
}

void Tensor::squeeze (int dim){
    if (dim < 0) dim = this->shape.size() + dim;
    assert (dim >= 0 && dim < this->shape.size());
    assert (this->shape[dim] == 1);
    std::new_shape(this->shape);
    new_shape.erase(new_shape.begin()+dim);
    this->shape_change(new_shape);
}

void Tensor::unsqueeze (int dim){
    if (dim < 0) dim = this->shape.size() + dim + 1;
    assert (dim >= 0 && dim <= this->shape.size());
    std::new_shape(this->shape);
    new_shape.insert(new_shape.begin() + dim, 1);
    this->shape_change(new_shape);

}

void Tensor::shape_change (std::vector<int> new_shape){
    int old_data_size = this->data_size;
    this->shape = new_shape;
    this->compute_data_size();
    assert (this->data_size == old_data_size)
    this->compute_stride();
}


Tensor* Tensor::deep_broadcast(std::vector<int> target_shape){
    assert (target_shape.size() >= this->shape.size());
    int dim_diff = target_shape.size() - this->shape.size();

    //lazy brodcast stride
    for (int i = 0; i < this->shape.size(); ++i){
        assert (this->shape[i] == 1 || this->shape[i] == target_shape[i+dim_diff]);
        if (this->shape[i] == 1) this->stride[i] =0;
    }

    Tensor* result = new Tensor (target_shape, this->requires_grad);
    for (int i = 0; i < result->shape_size; ++i){
        std::vector<int> indices = result->f_s(i);
        result->idx(indices) = this->idx(indices);
    }
    //back to correct stride
    this->compute_stride();

    return result;
}

void Tensor::init_grad (){
    assert (this->grad == nullptr);
    if (this->requires_grad == false) return;
    this->grad = new Tensor (this->shape,false);
}

void Tensor::topo_sort (std::set<Tensor*>& visited, std::vector<Tensor*>& sorted){
    if (visited.find(this) != visited.end()){
        return;
    }
    visited.insert(this);
    for (Tensor* operand : this->operands){
        operand->topo_sort(visited, sorted);
    }
    sorted.push_back(this);
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
        //reverse_sorted[i]->print();
        reverse_sorted[i]->backprop();
    }

    if (delete_intermediate == true){
        for (int i = 0; i < reverse_sorted.size(); ++i){
            if (reverse_sorted[i]->operands.size() > 0){
                delete reverse_sorted[i];
            }
        }
    }
}



Tensor* Tensor::add (Tensor* other){
    Tensor* result = nullptr;
    Op::add(this, other, result);
    result->operation = Operation::Add;
    result->operands.push_back(this);
    result->operands.push_back(other);

    return result;
}

Tensor* Tensor::minus (Tensor* other){
    Tensor* result = nullptr;
    Op::minus(this, other, result);

    result->operation = Operation::Minus;
    result->operands.push_back(this);
    result->operands.push_back(other);
    return result;
}

Tensor* Tensor::mul (Tensor* other){
    Tensor* result = nullptr;
    Op::mul(this, other, result);

    result->operation = Operation::Mul;
    result->operands.push_back(this);
    result->operands.push_back(other);
    return result;
}

Tensor* Tensor::div (Tensor* other){
    Tensor* result = nullptr;
    Op::div(this, other, result);

    result->operation = Operation::Div;
    result->operands.push_back(this);
    result->operands.push_back(other);
    return result;
}

Tensor* Tensor::pow (Tensor* other){
    Tensor* result = nullptr;
    Op::pow(this, other, result);

    result->operation = Operation::Pow;
    result->operands.push_back(this);
    result->operands.push_back(other);
    return result;
}

Tensor* Tensor::reduce_sum (std::vector<int> target_shape){
    Tensor* result = nullptr;
    Op::reduce_sum(this, target_shape, result);

    result->operation = Operation::ReduceSum;
    result->operands.push_back(this);
    return result;
}

Tensor* Tensor::compare (Tensor* other){
    Tensor* result = nullptr;
    Op::compare(this, other, result);
    return result;
}

Tensor* Tensor::max (Tensor* other){
    Tensor* result = nullptr;
    Op::max(this, other, result);

    result->operation = Operation::Max;
    result->operands.push_back(this);
    result->operands.push_back(other);
    return result;
}

Tensor* Tensor::min (Tensor* other){
    Tensor* result = nullptr;
    Op::min(this, other, result);

    result->operation = Operation::Min;
    result->operands.push_back(this);
    result->operands.push_back(other);
    return result;
}

Tensor* Tensor::matmul(Tensor* other){
    Tensor* result = nullptr;
    Op::matmul(this, other, result);

    result->operation = Operation::Matmul;
    result->operands.push_back(this);
    result->operands.push_back(other);
    return result;
}

Tensor* Tensor::slice (std::vector<std::vector<int>> slices){
    assert (slices.size() == this->shape.size());

    std::vector<int> result_shape (slices.size());
    for (int i = 0; i < slices.size(); ++i){
        assert (slices[i].size() == 0 || slices[i].size() == 2);
        if (slices[i].size() == 0){
            result_shape[i] = this->shape[i];
        }else{
            result_shape[i] = slices[i][1] - slices[i][0];
        }
    }
    Tensor* result = new Tensor (result_shape);

    for (int i = 0; i < result->data_size; ++i){
        std::vector<int> indices = result->f_s(i);
        for (int j = 0; j < indices.size(); ++j){
            if slices[j].size() != 0{
                indices[j] += slices[j][0];
            }
        }
        result[i] = this->idx(indices);
    }

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

Tensor* Tensor::reduce_sum(int target_dim){
    if (target_dim < 0) target_dim = this->shape.size() + target_dim;
    assert (this->shape.size() > target_dim && target_dim > 0);
    std::vector<int> result_shape = this->shape;
    result_shape[target_dim] = 1;
    return this->reduce_sum(result_shape);
}


void Tensor::copy (Tensor* target){
    assert (this != target);
    assert (this->requires_grad == target->requires_grad);
    this->shape = target->shape;
    this->stride = target->stride;
    delete this->data
    this->data = target->data;

}


void Tensor::add_ (Tensor* other) {Tensor* temp = this; Op::add(this, other, temp);};
void Tensor::minus_ (Tensor* other) {Tensor* temp = this; Op::minus (this, other, temp);};
void Tensor::mul_ (Tensor* other) {Tensor* temp = this;  Op::mul (this, other, temp);};
void Tensor::div_ (Tensor* other) {Tensor* temp = this; Op::div (this, other, temp);};


void Tensor::backprop (){

    switch (this->operation){
        case Operation::None:
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

    if (operand_A->requires_grad == true){
        Tensor* reduced_grad_A = nullptr;
        Op::reduce_sum(grad_A, operand_A->shape, reduced_grad_A);
        Op::add(operand_A->grad, reduced_grad_A, operand_A->grad);
        delete reduced_grad_A;
    }

    if (operand_B->requires_grad == true){
        Tensor* reduced_grad_B = nullptr;
        Op::reduce_sum(grad_B, operand_B->shape, reduced_grad_B);
        Op::add(operand_B->grad, reduced_grad_B, operand_B->grad);
        delete reduced_grad_B;
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
        Op::matmul (this->grad, operand_B->transpose().get(), grad_A);
    }
    if (operand_B->requires_grad == true){
        Op::matmul (operand_A->transpose().get(), this->grad, grad_B);
    }

    this->accumulate_grad(grad_A, grad_B);
    delete grad_A;
    delete grad_B;
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