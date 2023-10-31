#include <torch/torch.h>
#include <iostream>
#include <ros/ros.h>
#include <chrono>


#include "../include/net.h"
#include "../include/data_types.h"
#include "../include/replay_buffer.h"
#include "../include/train.h"
#include "../include/utils.h"


using namespace std;


int main() {
    Config config = _getParam();

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    cout << "Device : " << device << endl;
    
    PVNet pv(config);
    pv_->to(device);

    int n_batches = 256;

    torch::Tensor observations = torch::randn({n_batches, 105}).to(device);
    torch::Tensor target_policy = torch::nn::functional::one_hot(torch::randint(8, {n_batches, }), 9).to(device);
    torch::Tensor target_value = torch::randn({n_batches, 1}).to(device);

    Batch batch{os, target_policy, target_value};

    Loss loss = train(pv, batch);
    cout << loss.steer_loss << endl << loss.velocity_loss << endl << loss.value_loss << endl;

    return 0;
}

