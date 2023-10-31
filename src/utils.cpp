#include "../include/utils.h"



Config _getParam(ros::NodeHandle nh_) {
    Config config;

    nh_.getParam("DEBUG_MCTS", config.DEBUG_MCTS);
    nh_.getParam("c1", config.c1);
    nh_.getParam("c2", config.c2);
    nh_.getParam("angle_weight", config.angle_weight);
    nh_.getParam("MAX_DEPTH", config.MAX_DEPTH);
    nh_.getParam("MAX_ITER", config.MAX_ITER);
    nh_.getParam("MAX_TIME", config.MAX_TIME);
    nh_.getParam("dt", config.dt);
    nh_.getParam("gamma", config.gamma);

    nh_.getParam("action_space", config.action_space);
    nh_.getParam("observation_space", config.observation_space);

    nh_.getParam("base_hidden_state_1", config.base_hidden_state_1);
    nh_.getParam("base_hidden_state_2", config.base_hidden_state_2);
    nh_.getParam("base_output", config.base_output);
    nh_.getParam("policy_hidden_state_1", config.policy_hidden_state_1);
    nh_.getParam("policy_hidden_state_2", config.policy_hidden_state_2);
    nh_.getParam("value_hidden_state_1", config.value_hidden_state_1);
    nh_.getParam("value_hidden_state_2", config.value_hidden_state_2);
    
    nh_.getParam("DEBUG_Train", config.DEBUG_Train);
    nh_.getParam("lr", config.lr);
    nh_.getParam("policy_loss_weight", config.policy_loss_weight);
    nh_.getParam("value_loss_weight", config.value_loss_weight);
    nh_.getParam("train_interval", config.train_interval);
    nh_.getParam("l2_regularize_coefficent", config.l2_regularize_coefficent);
    nh_.getParam("batch_size", config.batch_size);
    nh_.getParam("uniform_policy_weight", config.uniform_policy_weight);


    nh_.getParam("angle_reward_weight", config.angle_reward_weight);
    nh_.getParam("terminal_reward_decay", config.terminal_reward_decay);

    nh_.getParam("enable_gpu", config.enable_gpu);

    nh_.getParam("load_model", config.load_model);
    nh_.getParam("load_dir", config.load_dir);
    nh_.getParam("save_interval", config.save_interval);
    nh_.getParam("save_dir", config.save_dir);

    nh_.getParam("DEBUG_Main", config.DEBUG_Main);


    return config;
}

vector<float> get_observation(State s, Coordinate g, vector<float> obs) {
    float x = s.pos.x;
    float y = s.pos.y;
    float th = s.pos.th;
    float s_ = s.s;
    float v = s.v;
    vector<float> observation{(float)((x - g.x + 16.0) / 32.0),
                              (float)((y - g.y + 30.0) / 60.0),
                              (float)((remainder((th - g.th), 2 * M_PI) + M_PI) / (2 * M_PI)),
                              (float)((s_ * RAD2DEG + 540.0) / (1080.0)), (float)((v * MPS2KMPH + 5.0) / (10.0))};
    assert(observation.size() == 5);
    if(obs.size() != 100) {
            cout << obs.size() << "\n" << obs[0];
    }
    assert(obs.size() == 100);
    vector<float> temp;
    for(int i = 0; i < obs.size(); i += 5) {
        temp.push_back((float)((obs[i] - x + 16.0) / 32.0));
        temp.push_back((float)((obs[i+1] - y + 30.0) / 60.0));
        temp.push_back((float)((remainder((obs[i+2] - th), 2 * M_PI) + M_PI) / (2 * M_PI)));
        temp.push_back(obs[i+3]);
        temp.push_back(obs[i+4]);
    }
    assert(temp.size() == 100);
    observation.insert(observation.end(), temp.begin(), temp.end());
    assert(observation.size() == 105);
    return observation;
}


float get_reward(Config config, State s, Coordinate g) {
    double x = s.pos.x - g.x;
    double y = s.pos.y - g.y;
    double th = s.pos.th - g.th;
    double dist = sqrt(pow(x, 2) + pow(y, 2));
    double c_ = config.angle_reward_weight;
    return 0.4 + 0.4 * (1 / (1 + c_) * func(dist) + c_ / (1 + c_) * func(abs(3 * th)));
}



////MCTS

double func(double x) {
    return 1 / (1.0 + x);
}

double Phi(double param) {
    if(param == 0.0) return 0;
    else return param + param / abs(param) * 0.8 * KMPH2MPS;
}

double inv_Phi(double v) {
    if(v == 0.0) return 0.0;
    else if(v > 0.0) return max(v - 0.8 * KMPH2MPS, 0.0);
    else return min(v + 0.8 * KMPH2MPS, 0.0);
}

tuple<double, double, double> localization(Coordinate car, Coordinate goal) {
    double tx = car.x - goal.x;
    double ty = car.y - goal.y;
    double x = tx * cos(goal.th) + ty * sin(goal.th);
    double y = -tx * sin(goal.th) + ty * cos(goal.th);
    double th = car.th - goal.th;
    return {x, y, th};
}