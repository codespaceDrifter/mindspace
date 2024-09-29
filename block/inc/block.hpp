#ifndef BLOCK_HPP
#define BLOCK_HPP

#include "tensor.hpp"

class Block {
public:

//training
virtual Tensor* forward(Tensor* input) = 0;

void set_mode_model(bool training);

~Block();
void delete_model();

void print(int tabs = 0);

std::vector<Tensor*> parameters;
std::vector<Block*> sub_blocks;
bool training = true; 

//save and load
virtual Block* factory_create () = 0;
virtual void init_members() = 0;


static std::map<std::string, Block*> block_types;

void register_block();

std::vector<Block*> get_all_blocks();
std::vector<Tensor*> get_all_tensors();
void set_IDs();
void ID_to_member(std::vector<Block*> all_blocks, std::vector<Tensor*> all_tensors);

void save (std::ofstream &out_file);
static Block* load (std::ifstream &in_file);
void save_model (std::string path = "bin/model.bin");
static Block* load_model (std::string path = "bin/model.bin");

std::string type;

int ID;
std::vector<int> parameters_ID;
std::vector<int> sub_blocks_ID;

/*
save load file format: (all info of a block should be stored in its code, subblocks, and tensors. additional member variables are not saved)
(int) total number of blocks
(int) total number of tensors
all the blocks
{
block id (int)
type size
type
sub blocks ids size
sub blocks ids
parameters ids size
parameters ids
}
all the tensors
{
tensor id (int)
shape size
shape
data size
data (float)
}

when loading:
read in the total number of blocks and tensors (two ints). create "all blocks" vector. loop through total block number of times calling "load", creating a custom typed block, push back.
do the same for tensors. except can create a vector of empty tensors. and call load on each one.
call ID to member on all the blocks
*/


};

#endif
