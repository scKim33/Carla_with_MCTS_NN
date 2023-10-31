#ifndef DATA_TYPES_H_
#define DATA_TYPES_H_

#include <torch/torch.h>
#include <random>

#include "config.h"


// Math Utils
#define DEG2RAD (M_PI / 180)
#define RAD2DEG 1 / DEG2RAD
#define MPS2KMPH 3.6
#define KMPH2MPS 1 / MPS2KMPH
#define DIST(x1, y1, x2, y2) sqrt((x1 - x2)*(x1 - x2) +(y1 - y2)*(y1 - y2))


using namespace std;


using PVNetInput = torch::Tensor;


struct PVNetOutput {
    torch::Tensor policy;
    torch::Tensor value;
};


struct Loss {
    torch::Tensor policy_loss;
    torch::Tensor value_loss;
};


struct Batch {
    vector<float> stacked_observations;
    vector<float> target_policies;
    vector<float> target_values;
    int n_samples;
    Config config;

    Batch(Config config_) {
        stacked_observations.reserve(config_.observation_space * 1e7);
        target_policies.reserve(config_.action_space * 1e7);
        target_values.reserve(1e7);
        config = config_;
        n_samples = 0;
    }
    Batch(
        Config config_, vector<float> os, vector<float> tp, vector<float> tval)
        : stacked_observations(os), target_policies(tp), target_values(tval) {
        stacked_observations.reserve(config_.observation_space * 1e7);
        target_policies.reserve(config_.action_space * 1e7);
        target_values.reserve(1e7);
        config = config_;
        n_samples = target_values.size();
        }


    Batch &operator+=(const Batch &other) {
        stacked_observations.reserve(stacked_observations.size() + other.stacked_observations.size());
        stacked_observations.insert(stacked_observations.end(), other.stacked_observations.begin(), other.stacked_observations.end());
        target_policies.reserve(target_policies.size() + other.target_policies.size());
        target_policies.insert(target_policies.end(), other.target_policies.begin(), other.target_policies.end());
        target_values.reserve(target_values.size() + other.target_values.size());
        target_values.insert(target_values.end(), other.target_values.begin(), other.target_values.end());
        n_samples += other.n_samples;
        return *this;
    }


    void print_sample() {
        if(n_samples == 0) {
            cout << "No data in the batch\n";
        } else {
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<int> dis(0, n_samples - 1);
            int r = dis(gen);
            vector<double> os_(stacked_observations.begin() + config.observation_space * r, stacked_observations.begin() + config.observation_space * r + config.observation_space);
            // vector<double> s_(b.steer_cmds.begin() + 5*r, b.steer_cmds.begin() + 5*r + 5);
            // vector<double> v_(b.velocity_cmds.begin() + 2*r, b.velocity_cmds.begin() + 2*r + 2);
            vector<double> tp_(target_policies.begin() + config.action_space * r, target_policies.begin() + config.action_space * r + config.action_space);
            vector<double> tv_(target_values.begin() + r, target_values.begin() + r + 1);
            cout << "Batch Info\n"
            << "Batch size : " << n_samples << "\n"
            << "Print example of element (element " << r << ")\n"
            << "Observations : " << os_ << "\n"
            // << "Rewards : " << b.values[r] << endl
            // << "Actions : " << s_ << ", " << v_ << endl
            << "Target Policy : " << tp_ << "\n"
            << "Target Value : " << tv_ << "\n";
        }
    }


    void clear() {
        stacked_observations.clear();
        target_policies.clear();
        target_values.clear();
        n_samples = 0;
    }
};


// MCTS
struct Coordinate {
    double x; // m
    double y; // m
    double th; // rad

    Coordinate() {} 
    Coordinate(double x_, double y_, double th_) : x(x_), y(y_), th(th_) {} 
    Coordinate operator-(const Coordinate &c) const {
        return Coordinate(x - c.x, y - c.y, th - c.th);
    }
};


// MCTS
struct State {
    Coordinate pos;
    double v; // m/s
    double s; // rad
    double dir; // {-1, +1}

    friend ostream& operator<<(ostream &o, const State &s) {
        o << "x: " << s.pos.x << ", y: " << s.pos.y << ", th: " << s.pos.th * RAD2DEG << ", v: " << s.v * s.dir << ", s: " << s.s;
        return o;
    }
};

#endif