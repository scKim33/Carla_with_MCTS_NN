#include <torch/torch.h>
#include <iostream>
#include <ros/ros.h>
#include <chrono>


#include "../include/net.h"
#include "../include/data_types.h"
#include "../include/replay_buffer.h"
#include "../include/train.h"
#include "../include/utils.h"
#include "../include/config.h"


using namespace std;


int main() {
    int argc = 0;
    char** argv;
    ros::init(argc, argv, "MCTS_node");
    ros::NodeHandle nh_;
    ros::NodeHandle priv_nh("~");

    Config config = _getParam(nh_);
    torch:: TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);


    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    cout << "Device : " << device << endl;
    
    device = torch::kCPU;
    PVNet pv(config);
    pv->to(device);

    int n_batches = 256;

    torch::Tensor observations = torch::randn({n_batches, 105});
    // torch::Tensor indices = torch::randint(9, {n_batches, }).to(torch::kInt32);
    // torch::Tensor target_policy = torch::nn::functional::one_hot(indices, 9);
    torch::Tensor target_policy = torch::randn({n_batches, 9});
    torch::Tensor target_value = torch::randn({n_batches, 1});
    vector<float> observations_(observations.data_ptr<float>(), observations.data_ptr<float>() + observations.numel());
    vector<float> target_policy_(target_policy.data_ptr<float>(), target_policy.data_ptr<float>() + target_policy.numel());
    vector<float> target_value_(target_value.data_ptr<float>(), target_value.data_ptr<float>() + target_value.numel());

    Batch batch(config, observations_, target_policy_, target_value_);

    Loss loss = train(config, pv, batch, options);
    cout << loss.policy_loss << endl << loss.value_loss << endl;

    return 0;
}

