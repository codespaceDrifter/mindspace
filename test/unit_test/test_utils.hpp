#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP


std::string bool_to_str (bool pass){
    if (pass == true) return "PASSED";
    else return "FAILED";
}

std::string vec_to_str (std::vector<int> vec){
    std::string result;
    for (int i = 0; i < vec.size(); ++i){
        result += std::to_string(vec[i]);
        if (i != vec.size() -1 ) result += ", ";
    }
    result = "{" + result + "}";
    return result;
}

void check(bool temp_success, bool& all_success){
    if (temp_success == false) all_success = temp_success;
}

bool appro_equal (float a, float b, float episilon = 1e-5){
    float diff = a-b;
    if (diff < 0) diff = -diff;
    return diff < episilon;
}

bool all_equal (TensorCore* a, float value){
    bool success = true;
    for (int i = 0; i < a->data_size; ++i){
        check (appro_equal(a->data[i], value),success);
    }
    return success;
}

bool vec_all_equal (TensorCore*a, std::vector<float> b){
    bool success = true;
    for (int i = 0; i < a->data_size; ++i){
        check (appro_equal(a->data[i], b[i]), success);
    }
    return success;
}




#endif