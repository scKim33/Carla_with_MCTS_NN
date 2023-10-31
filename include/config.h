#ifndef CONFIG_H_
#define CONFIG_H_

#include<string>

using namespace std;

struct Config {
    // DEBUG parameters
    bool DEBUG_MCTS;
    bool DEBUG_Train;
    bool DEBUG_Main;
    
    // MCTS parameters
    double c1; // parameter of PUCT
    double c2;
    double angle_weight; // parameter of reward function, reward weight between angle and euclidian
    int MAX_DEPTH; // maximum depth of MCTS Tree 
    int MAX_ITER; // maximum backups of MCTS Tree
    double MAX_TIME; // MCTS process terminate within max time
    double dt; // time steps interval between nodes
    double gamma; // n-step reward decay


    // NN parameters

    // dim
    int action_space;
    int observation_space;

    // net
    int base_hidden_state_1;
    int base_hidden_state_2;
    int base_output;
    int policy_hidden_state_1;
    int policy_hidden_state_2;
    int value_hidden_state_1;
    int value_hidden_state_2;

    // train
    float lr;
    float policy_loss_weight;
    float value_loss_weight;
    int train_interval;
    float l2_regularize_coefficent;
    int batch_size;
    float uniform_policy_weight;

    // reward
    float angle_reward_weight;
    float terminal_reward_decay;

    // device
    bool enable_gpu;
    
    // save and load
    bool load_model;
    string load_dir;
    int save_interval;
    string save_dir;


    // Main parameters
};

#endif