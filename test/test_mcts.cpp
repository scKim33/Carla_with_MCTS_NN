#include <torch/torch.h>
#include <iostream>
#include <ros/ros.h>


#include "../include/net.h"
#include "../include/data_types.h"
#include "../include/replay_buffer.h"
#include "../include/mcts.h"


using namespace std;


int main() {
    int argc = 0;
    char** argv;
    ros::init(argc, argv, "MCTS_node");
    ros::NodeHandle nh_;
    ros::NodeHandle priv_nh("~");

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    cout << "Device : " << device << endl;

    PVNet pv(65, 100, 100, 100, 100, 100, 100, 100, 100, 100);
    pv->to(torch::kCPU);

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
    // for(int i = 0; i < 20; i++) {
    //     Coordinate obs{1.0  + 1.0 * i, 2.0 + 2.0 * i, 3.0 + 3.0 * i};
    //     obstacles.push_back(obs);
    // }

    // Sub_ObstaclePoses = nh_.subscribe("/obs_poses", 1, &Callback_ObstaclePoses);
    // Sub_GoalPosition = nh_.subscribe("/parking_cands", 1, &Callback_Goal);
    nav_msgs::Path mcts_path;
    vector<float> policy;
    MCTS mcts_agent(nh_, state, goal, obstacles, pv);
    tie(state.s, state.v, policy, mcts_path) = mcts_agent.main();
    
    // Pub_MctsPath.publish(mcts_path);

    return 0;
}

