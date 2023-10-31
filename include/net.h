#ifndef NET_H_
#define NET_H_

#include <torch/torch.h>

#include "data_types.h"
#include "config.h"


struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M);
  torch::Tensor forward(torch::Tensor input);
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};


class Base_MLPImpl : public torch::nn::Module {
 public:
    Base_MLPImpl(int64_t input_size, int64_t hidden_size1, int64_t hidden_size2, int64_t output_size);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layers;
};
TORCH_MODULE(Base_MLP);


class Head_MLPImpl : public torch::nn::Module {
 public:
    Head_MLPImpl(int64_t input_size, int64_t hidden_size1, int64_t hidden_size2, int64_t output_size);
    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Sequential layers;
};
TORCH_MODULE(Head_MLP);



class PVNetImpl : public torch::nn::Module {
 public:
    PVNetImpl(Config config);

    PVNetOutput forward(torch::Tensor x);

 private:
    Base_MLP base;
    Head_MLP policy_head;
    Head_MLP value_head;
};
TORCH_MODULE(PVNet);


#endif