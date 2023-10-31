#include "../include/train.h"



Loss train(Config config, PVNet& net, Batch& batch, torch::TensorOptions tensor_options) {
    bool DEBUG = config.DEBUG_Train;

    torch::Tensor stacked_observations =
        torch::from_blob(batch.stacked_observations.data(), {batch.n_samples, config.observation_space}, tensor_options);
    torch::Tensor target_policies = 
        torch::from_blob(batch.target_policies.data(), {batch.n_samples, config.action_space}, tensor_options);
    torch::Tensor target_values = 
        torch::from_blob(batch.target_values.data(), {batch.n_samples, 1}, tensor_options);
    
    PVNetOutput out = net->forward(stacked_observations);

    auto mseloss = torch::nn::MSELoss();
    auto celoss = torch::nn::CrossEntropyLoss();
    torch::Tensor policy_loss = celoss(target_policies, out.policy);
    torch::Tensor value_loss = mseloss(target_values, out.value);

    if(DEBUG) {
        cout << "Policy Loss : " << policy_loss.item<float>() << ", " << "Value Loss : " <<  config.value_loss_weight * value_loss.item<float>() << "\n";
    }

    torch::Tensor total_loss = policy_loss + config.value_loss_weight * value_loss;

    torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(config.lr).betas(std::make_tuple(0.5, 0.5)).weight_decay(config.l2_regularize_coefficent));

    net->zero_grad();
    total_loss.backward();
    optimizer.step();

    return {policy_loss, value_loss};
}