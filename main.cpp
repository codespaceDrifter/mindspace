#include "all_tests.hpp"
#include "all_util_inc.hpp"

int main (){

    //run_all_tests();
    Tensor* input = new Tensor (2,3);
    input->arrange(0,1);
    Tensor* target = new Tensor (2,3);
    target->arrange(1,2);

    TensorDataset* tensor_dataset = new TensorDataset(input, target);
    DataLoader* dataloader = new DataLoader(tensor_dataset, 1, false);
    int epoch = 1'00;

    Block* DenseModel = new DenseLayer(3,3);
    DenseModel -> init_randomize_model(-1,1);
    StandardOptimizer* opt = new StandardOptimizer(DenseModel, 0.01, 0, 1);

    for (int i = 0; i < epoch; ++i){
        std::vector<Tensor*> batch = dataloader->getNextBatch();
        Tensor* input = batch[0];
        Tensor* target = batch[1];
        Tensor* pred = DenseModel->forward(input);
        Tensor* loss = MSEloss(pred, target);

        std::cout << "loss: " <<std::endl;
        loss->print();

        loss->backward_model();
        opt->step();
        opt->zero_grad();
    }


    return 0;
}