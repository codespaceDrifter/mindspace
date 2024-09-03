#include "tensor_core.hpp"


TensorCore* TensorCore::createTensorCore(std::vector<int> shape){
    bool use_cpu = true;
    if (use_cpu){
        return new CpuTensorCore(shape);
    }
    return nullptr;
}


TensorCore::TensorCore (std::vector<int> shape){
    this->owns_data = true;
    this->data_size = 0;
    this->data = new float[0];
    this->shape = shape;
    this->shape_update();
}

TensorCore::TensorCore(std::vector<float> data, std::vector<int> shape) : TensorCore (shape) {
    assert (data.size() == this->data_size);
    for (int i = 0; i < data_size; ++i){
        this->data[i] = data[i];
    }
}

TensorCore::~TensorCore(){
    if (this->owns_data == true) delete[] this->data;
}


void TensorCore::shape_update(){
    //data size
    int new_data_size = std::min(1, static_cast<int>(this->shape.size()));
    for (int i = 0; i < this->shape.size(); ++i){
        new_data_size *= this->shape[i];
    }
    this->data_resize(new_data_size);
    this->data_size = new_data_size;

    //stride
    this->stride.clear();
    int cur_stride = 1;
    for (int i = this->shape.size()-1 ; i >= 0; --i){
        if (this->shape[i] == 1){
            this->stride.insert(stride.begin(), 0);
        } else{
            this->stride.insert(stride.begin(), cur_stride);
        }
        cur_stride *= this->shape[i];
    }
}



void TensorCore::data_resize(int new_size){
    if (this->owns_data == false) return;
    assert (new_size >= this->data_size);

    float* new_data = new float[new_size];
    for (int i = 0; i < new_size; ++i){
        if (i < this->data_size) new_data[i] = data[i];
        else new_data[i] = 0;
    }
    this->data_size = new_size;
    delete[] this->data;
    this->data = new_data;
}


void TensorCore::data_switch (float* new_data, int new_size){
    if (this->owns_data == false) return;
    delete[] this->data;
    this->data = new_data;
    if (new_size != -1) this->data_size = new_size;
}



void TensorCore::randomize (float low, float high){
    for (int i = 0; i< this->data_size; ++i){
        float cur = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * (high - low) +low;
        this->data[i] = cur;
    }
}

void TensorCore::fill (float value){
    for (int i = 0; i < this->data_size; ++i){
        this->data[i] = value;
    }
}

void TensorCore::arrange (int start){
    for (int i = 0; i < this->data_size; ++i){
        this->data[i] = start + i;
    }
}

void TensorCore::print (){
    std::cout << "shape: (";
    for (int i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i != shape.size() - 1) std::cout << ", ";
    }
    std::cout <<")"<<std::endl;
    std::cout <<"data: "<<std::endl;
    std::cout <<this->to_string();
}

std::string TensorCore::to_string(){
    if (this->shape.size() == 0 ) return "";

    std::vector<std::string> line_vec;
    std::string temp_str;

    for (int i = 0; i < this->data_size; ++i){
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


std::unique_ptr<TensorCore> TensorCore::create_shallow_equal (){
    TensorCore* raw_result = TensorCore::createTensorCore();

    delete[] raw_result -> data;
    raw_result -> data = this -> data;
    raw_result -> owns_data = false;
    raw_result -> data_size = this->data_size;
    raw_result ->shape = this -> shape;
    raw_result ->stride = this -> stride;

    std::unique_ptr<TensorCore> unique_result (raw_result);
    return unique_result;
}

std::unique_ptr<TensorCore> TensorCore::create_deep_equal (){
    TensorCore* raw_result = TensorCore::createTensorCore(this->shape);
    for (int i = 0; i < this->data_size; ++i){
        raw_result->data[i] = this->data[i];
    }
    std::unique_ptr<TensorCore> result (raw_result);
    return result;
}

std::unique_ptr<TensorCore> TensorCore::transpose(int dim1, int dim2){
    if (dim1 < 0) dim1 = this->shape.size() + dim1;
    if (dim2 < 0) dim2 = this->shape.size() + dim2;
    std::unique_ptr<TensorCore> result = this->create_shallow_equal(); 
    std::swap(result->shape[dim1], result->shape[dim2]);
    std::swap (result->stride[dim1], result->stride[dim2]);
    return result; 
}

void TensorCore::transpose_ (int dim1, int dim2){
    std::unique_ptr<TensorCore> temp_transposed = this->transpose();
    float* new_data = new float[this->data_size];
    for (int i = 0; i < this->data_size; ++i){
        new_data[i] = temp_transposed->idx(temp_transposed->f_s(i));
    }
    this->data_switch(new_data);
    this->shape = temp_transposed->shape;
    this->shape_update();
}


std::unique_ptr<TensorCore> TensorCore::squeeze(int dim) {
    if (dim < 0) dim = this->shape.size() + dim;
    std::unique_ptr<TensorCore> result = this->create_shallow_equal();
    if (result->shape[dim] == 1){
        result->shape.erase(result->shape.begin() + dim);
        result->stride.erase(result->stride.begin() + dim);
    }
    return result;
}

void TensorCore::squeeze_ (int dim){
    if (dim < 0) dim = this->shape.size() + dim;
    if (this->shape[dim] == 1){
        this->shape.erase(this->shape.begin() + dim);
        this->shape_update();
    }
}



std::unique_ptr<TensorCore> TensorCore::unsqueeze(int dim){
    std::unique_ptr<TensorCore> result = this->create_shallow_equal();
    if (dim < 0) dim = this->shape.size() + dim + 1;
    result->shape.insert(result->shape.begin()+dim, 1);
    result->stride.insert(result->stride.begin() + dim, 0);
    return result;
}

void TensorCore::unsqueeze_ (int dim){
    if (dim < 0) dim = this->shape.size() + dim + 1;
    this->shape.insert(this->shape.begin() + dim, 1);
    this->shape_update();
}



std::vector<int> TensorCore::max_broadcast_shape (TensorCore* other) const{
    std::vector<int> this_padded = this->shape;
    std::vector<int> other_padded = other->shape;
    this_padded.insert(this_padded.begin(), std::max(0, static_cast<int>(other_padded.size() - this_padded.size()) ), 1);
    other_padded.insert(other_padded.begin(), std::max(0, static_cast<int>(this_padded.size() - other_padded.size())),1);
    std::vector<int> result;
    for (int i = 0; i < this_padded.size(); ++i){
        result.push_back(std::max(this_padded[i], other_padded[i]));
    }
    return result;
}

bool TensorCore::element_op_viable(TensorCore* other) const{
    std::vector<int> this_padded = this->shape;
    std::vector<int> other_padded = other->shape;
    this_padded.insert(this_padded.begin(), std::max(0,static_cast<int>(other_padded.size() - this_padded.size())), 1);
    other_padded.insert(other_padded.begin(), std::max(0,static_cast<int>(this_padded.size() - other_padded.size())),1);
    for (int i = 0; i < this_padded.size(); ++i){
        if (this_padded[i] != other_padded[i] && this_padded[i] != 1 && other_padded[i] != 1) return false;
    }
    return true;
}

bool TensorCore::matmul_viable(TensorCore* other) const{
    std::vector<int> this_padded = this->shape;
    std::vector<int> other_padded = other->shape;
    this_padded.insert(this_padded.begin(), std::max(0,static_cast<int>(other_padded.size() - this_padded.size())), 1);
    other_padded.insert(other_padded.begin(), std::max(0,static_cast<int>(this_padded.size() - other_padded.size())),1);
    for (int i = 0; i < this_padded.size() - 2; ++i){
        if (this_padded[i] != other_padded[i] && this_padded[i] != 1 && other_padded[i] != 1) return false;
    }
    if (this_padded[this_padded.size()-1] != other_padded[other_padded.size()-2]) return false;
    return true;
}