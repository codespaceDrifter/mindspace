#ifndef POS_EMBED_LAYER
#define POS_EMBED_LAYER


#include "block.hpp"

class PosEmbedLayer : public Block {
public:
PosEmbedLayer (int seq_len = 0, int embed_dim = 0);

Tensor* forward(Tensor* input) override;
void init_members() override;

Tensor* mask;

};





#endif