#ifndef STANDARD_OPTIMIZER_HPP
#define STANDARD_OPTIMIZER_HPP

#include "optimizer.hpp"

class StandardOptimizer : public Optimizer{
public:
    StandardOptimizer (Block* model, float learning_rate, float weight_decay, float grad_limit);

    void step() override;
};

#endif
