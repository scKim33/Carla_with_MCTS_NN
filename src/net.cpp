#include "../include/net.h"

Net::Net(int64_t N, int64_t M) : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
}


torch::Tensor Net::forward(torch::Tensor input) {
    return linear(input) + another_bias;
}


Base_MLPImpl::Base_MLPImpl(int64_t input_size, int64_t hidden_size1, int64_t hidden_size2, int64_t output_size) {
    layers->push_back("linear1", torch::nn::Linear(input_size, hidden_size1));
    layers->push_back("activation1", torch::nn::ReLU());
    layers->push_back("linear2", torch::nn::Linear(hidden_size1, hidden_size2));
    layers->push_back("activation2", torch::nn::ReLU());
    layers->push_back("linear3", torch::nn::Linear(hidden_size2, output_size));
    register_module("Base_MLP", layers);
}

torch::Tensor Base_MLPImpl::forward(torch::Tensor x) {
    torch::Tensor out = layers->forward(x);
    return out;
}


Head_MLPImpl::Head_MLPImpl(int64_t input_size, int64_t hidden_size1, int64_t hidden_size2, int64_t output_size) {
    layers->push_back("linear1", torch::nn::Linear(input_size, hidden_size1));
    layers->push_back("activation1", torch::nn::ReLU());
    layers->push_back("linear2", torch::nn::Linear(hidden_size1, hidden_size2));
    layers->push_back("activation2", torch::nn::ReLU());
    layers->push_back("linear3", torch::nn::Linear(hidden_size2, output_size));
    layers->push_back("sigmoid1", torch::nn::Sigmoid());
    register_module("Head_MLP", layers);
}

torch::Tensor Head_MLPImpl::forward(torch::Tensor x) {
    torch::Tensor out = layers->forward(x);
    return out;
}


PVNetImpl::PVNetImpl(Config config) : 
      base(config.observation_space, config.base_hidden_state_1, config.base_hidden_state_2, config.base_output),
      policy_head(config.base_output, config.policy_hidden_state_1, config.policy_hidden_state_2, config.action_space),
      value_head(config.base_output, config.value_hidden_state_1, config.value_hidden_state_2, 1) {
        register_module("base", base);
        register_module("policy_head", policy_head);
        register_module("value_head", value_head);
}


PVNetOutput PVNetImpl::forward(torch::Tensor x) {
    torch::Tensor out = base->forward(x);
    torch::Tensor out_p = policy_head->forward(out);
    torch::Tensor out_v = value_head->forward(out);
    return {out_p, out_v};
}