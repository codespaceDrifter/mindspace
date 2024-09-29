#include "block.hpp"


Block::~Block(){
    for (Tensor* tensor : this->parameters){
        delete tensor;
    }
}

void Block::print(int tabs){
    std::string tabs_str = std::string (tabs, '\t');
    std::cout<<tabs_str<<this->type<<std::endl;
    std::cout<<tabs_str<<"parameters: ";


    for (int i = 0; i < this->parameters.size(); ++i){
        std::cout<<Tensor::vec_str(this->parameters[i]->shape);
        if (i != this->parameters.size() - 1) std::cout<<",    ";
    }
    std::cout<<"\n"<<"\n";
    tabs ++;
    for (Block* block : this->sub_blocks){
        block->print(tabs);
    }
}

void Block::set_mode_model(bool training){
    std::vector<Block*> all_blocks = this->get_all_blocks();
    for (Block* block : all_blocks){
        block->training = training;
    }
}

void Block::delete_model(){
    std::vector<Block*> all_blocks = this->get_all_blocks();
    for (int i = 1; i < all_blocks.size(); ++i){
        delete all_blocks[i];
    }
    delete this;
}


std::map<std::string, Block*> Block::block_types;

void Block::register_block(){
    if (Block::block_types.find (this->type) == Block::block_types.end()){
        Block::block_types[this->type] = nullptr;
        Block::block_types[this->type] = this->factory_create();
    }
}



std::vector<Block*> Block::get_all_blocks(){
    std::vector<Block*> all_blocks;
    std::queue<Block*> q;
    q.push(this);
    while (!q.empty()){
        Block* cur_block = q.front();
        q.pop();
        all_blocks.push_back(cur_block);
        for (Block* sub : cur_block->sub_blocks){
            q.push(sub);
        }
    }
    return all_blocks;
}

std::vector<Tensor*> Block::get_all_tensors(){
    std::vector<Block*> all_blocks = this->get_all_blocks();
    std::vector<Tensor*> all_tensors;
    for (Block* block : all_blocks){
        for (Tensor* tensor : block->parameters){
            all_tensors.push_back(tensor);
        }
    }
    return all_tensors;
}

void Block::set_IDs(){
    std::vector<Block*> all_blocks = this->get_all_blocks();
    std::vector<Tensor*> all_tensors = this->get_all_tensors();
    for (int i = 0; i < all_blocks.size(); ++i){
        all_blocks[i] ->ID = i;
    }
    for (int i = 0; i < all_tensors.size(); ++i){
        all_tensors[i] ->ID = i;
    }

    for (Block* block: all_blocks){
        block->parameters_ID.clear();
        block->sub_blocks_ID.clear();

        for (Tensor* tensor : block->parameters){
            block->parameters_ID.push_back(tensor->ID);
        }
        for (Block* sub_block : block->sub_blocks){
            block->sub_blocks_ID.push_back(sub_block->ID);
        }
    }
}

void Block::ID_to_member(std::vector<Block*> all_blocks, std::vector<Tensor*> all_tensors){

    for (Tensor* tensor : this->parameters){
        delete tensor;
    }
    for (Block*  block: this->sub_blocks){
        delete block;
    }

    this->sub_blocks.clear();
    this->parameters.clear();


    for (int tensor_id : this->parameters_ID){
        for (Tensor* tensor : all_tensors){
            if (tensor->ID == tensor_id){
                    this->parameters.push_back(tensor);
            }
        }
    }

    for (int block_id : this->sub_blocks_ID){
        for (Block* block : all_blocks){
            if (block->ID == block_id){
                    this->sub_blocks.push_back(block);
            }
        }
    }
    assert (this->sub_blocks_ID.size() == this->sub_blocks.size());
    assert (this->parameters_ID.size() == this->parameters.size());
}

void Block::save (std::ofstream &out_file){
    if (!out_file) {
        std::cout << "Invalid ofstream object" << std::endl;
        return;
    }
    out_file.seekp(0, std::ios::end);
    out_file.write (reinterpret_cast<const char*> (&this->ID), sizeof(this->ID));
    int type_size = this->type.size();
    out_file.write (reinterpret_cast<const char*> (&type_size), sizeof (type_size));
    out_file.write (this->type.c_str(), type_size);
    int sub_blocks_ID_size = this->sub_blocks_ID.size();
    out_file.write (reinterpret_cast<const char*> (&sub_blocks_ID_size), sizeof (sub_blocks_ID_size));
    out_file.write (reinterpret_cast<const char*> (this->sub_blocks_ID.data()), sub_blocks_ID_size * sizeof(int));
    int parameters_ID_size = this->parameters_ID.size();
    out_file.write (reinterpret_cast<const char*> (&parameters_ID_size), sizeof (parameters_ID_size));
    out_file.write (reinterpret_cast<const char*> (this->parameters_ID.data()), parameters_ID_size * sizeof(int));
}

Block* Block::load (std::ifstream &in_file){
    if (!in_file) {
        std::cout << "Invalid ifstream object" << std::endl;
        return nullptr;
    }

    int ID;
    in_file.read(reinterpret_cast<char*> (&ID), sizeof(ID));
    std::string type;
    int type_size;
    in_file.read (reinterpret_cast<char*> (&type_size), sizeof(type_size));
    type.resize(type_size);
    in_file.read (reinterpret_cast<char*> (&type[0]), type_size);

    Block * result = Block::block_types[type]->factory_create();
    result->ID = ID;
    int sub_blocks_ID_size;
    in_file.read (reinterpret_cast<char*> (&sub_blocks_ID_size), sizeof(sub_blocks_ID_size));
    result->sub_blocks_ID.resize(sub_blocks_ID_size);
    in_file.read (reinterpret_cast<char*> (result->sub_blocks_ID.data()), sub_blocks_ID_size * sizeof (int));
    int parameters_ID_size;
    in_file.read (reinterpret_cast<char*> (&parameters_ID_size), sizeof(parameters_ID_size));
    result->parameters_ID.resize(parameters_ID_size);
    in_file.read (reinterpret_cast<char*> (result->parameters_ID.data()), parameters_ID_size * sizeof (int));

    return result;
}

void Block::save_model (std::string path){
    std::ofstream out_file(path, std::ios::binary);
    if (!out_file){
        std::cout << "Invalid ofstream object" << std::endl;
        return;
    }
    this->set_IDs();
    std::vector<Block*> all_blocks = this->get_all_blocks();
    std::vector<Tensor*> all_tensors = this->get_all_tensors();
    int total_num_blocks = all_blocks.size();
    int total_num_tensors = all_tensors.size();
    out_file.write(reinterpret_cast<const char*> (& total_num_blocks), sizeof (total_num_blocks));
    out_file.write(reinterpret_cast<const char*> (& total_num_tensors), sizeof (total_num_tensors));
    for (Block* block : all_blocks){
        block->save(out_file);
    }
    for (Tensor* tensor: all_tensors){
        tensor->save(out_file);
    }
}


Block* Block::load_model (std::string path){
    std::ifstream in_file (path, std::ios::binary);
    if (!in_file){
        std::cout << "Invalid ifstream object" <<std::endl;
        return nullptr;
    }
    int total_num_blocks;
    int total_num_tensors;
    in_file.read (reinterpret_cast<char*> (& total_num_blocks), sizeof (total_num_blocks));
    in_file.read (reinterpret_cast<char*> (& total_num_tensors), sizeof (total_num_tensors));
    std::vector<Block*> all_blocks;
    std::vector<Tensor*> all_tensors;

    for (int i = 0; i < total_num_blocks; ++i){
        Block* cur_block = Block::load(in_file);
        all_blocks.push_back(cur_block);
    }
    for (int i = 0; i < total_num_tensors; ++i){
        Tensor* cur_tensor = new Tensor();
        cur_tensor->load(in_file);
        all_tensors.push_back(cur_tensor);
    }

    for (Block* block : all_blocks){
        block->ID_to_member(all_blocks,all_tensors);
        block->init_members();
    }

    return all_blocks[0];
}