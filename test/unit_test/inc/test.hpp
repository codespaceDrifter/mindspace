#ifndef TEST_HPP
#define TEST_HPP

#include "tensor.hpp"
#include "operations.hpp"

class Test{
public:

Test(){
    this->all_passed = true;
    this->debug = false;
};

virtual void run_tests() = 0;

void check_all_passed (bool pass){
    if (pass == false) this->all_passed = false;
}

static std::string bool_to_str (bool pass){
    if (pass == true) return "PASSED\n";
    else return "FAILED\n";
}

static void check (bool pass, bool & success){
    if (pass == false) success = false;
}


static bool appro_equal (float a, float b, float episilon = 1e-3){
    float diff = a-b;
    if (diff < 0) diff = -diff;
    return diff < episilon;
}

static bool all_equal (Tensor* a, float value){
    bool success = true;
    for (int i = 0; i < a->shape_size; ++i){
        check (appro_equal(a->data[i], value),success);
    }
    return success;
}

static bool all_equal (Tensor*a, std::vector<float> b){
    bool success = true;
    for (int i = 0; i < a->shape_size; ++i){
        check (appro_equal(a->idx(a->f_s(i)), b[i]), success);
    }
    return success;
}

static bool all_equal (Tensor*a, std::initializer_list<float> b){ return all_equal(a, std::vector<float> (b));};

static bool all_equal (std::vector<int> a, std::vector<int> b){
    bool success = true;
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); ++i){
        check (a[i] == b[i], success);
    }
    return success;
}

static bool all_equal (std::vector<int> a, std::initializer_list<int> b){return all_equal (a, std::vector<int> (b));};

static bool all_equal (Tensor* a, Tensor* b){
    bool success = true;
    if (a->shape_size != b->shape_size) return false;
    for (int i = 0; i < a->shape_size; ++i){
        check (appro_equal(a->idx(a->f_s(i)), b->idx(b->f_s(i))),success);
    }
    return success;
}

static bool all_equal (std::unique_ptr<Tensor>& a, Tensor* b){return all_equal (a.get(), b);}

static bool all_equal (Tensor* a, std::unique_ptr<Tensor>& b){return all_equal (a, b.get());}

static bool all_equal (std::unique_ptr<Tensor>& a, std::unique_ptr<Tensor>& b){return all_equal (a.get(), b.get());}


bool all_passed;
bool debug;
};




#endif