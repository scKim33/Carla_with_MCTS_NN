#include <torch/torch.h>
#include <iostream>
#include <ros/ros.h>


#include "../include/net.h"
#include "../include/data_types.h"
#include "../include/replay_buffer.h"
#include "../include/mcts.h"
#include "../include/config.h"


using namespace std;


int main() {
    int argc = 0;
    char** argv;
    ros::init(argc, argv, "MCTS_node");
    ros::NodeHandle nh_;
    ros::NodeHandle priv_nh("~");

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    cout << "Device : " << device << endl;

    Config config = _getParam(nh_);
    torch:: TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    device = torch::kCPU;

    PVNet pv(config);
    pv->to(device);

    // vector<Coordinate> obstacles;
    vector<float> obstacles;
    Coordinate goal;
    Coordinate car;
    State state;

    state.pos.x = 1.0;
    state.pos.y = 2.0;
    state.pos.th = 30 * DEG2RAD;
    state.s = 300 * DEG2RAD;
    state.v = 1.0;
    goal.x = 2.0;
    goal.y = 3.0;
    goal.th = 50 * DEG2RAD;
    for(int i = 0; i < 100; i++) {
        obstacles.push_back(i);
    }

    // Sub_ObstaclePoses = nh_.subscribe("/obs_poses", 1, &Callback_ObstaclePoses);
    // Sub_GoalPosition = nh_.subscribe("/parking_cands", 1, &Callback_Goal);
    nav_msgs::Path mcts_path;
    vector<float> policy;
    int backup_count;
    MCTS mcts_agent(config, nh_, state, goal, obstacles, pv, options);
    tie(state.s, state.v, policy, mcts_path, backup_count) = mcts_agent.main();
    cout << state.s << ", " << state.v << "\n" << policy << "\n";
    // Pub_MctsPath.publish(mcts_path);

    return 0;
}

