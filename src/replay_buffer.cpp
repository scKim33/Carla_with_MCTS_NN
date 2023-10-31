#include "../include/replay_buffer.h"


Replay_buffer::Replay_buffer(Config config_) : batch(config_) {
    config = config_;
}


Replay_buffer::~Replay_buffer() {}


Batch Replay_buffer::sample(int batch_size) {
    Batch samples(config);
    if(batch.n_samples < batch_size) {
        batch_size = batch.n_samples;
    }

    vector<int> idx(batch.n_samples);
    std::iota(idx.begin(), idx.end(), 0);
    std::random_shuffle(idx.begin(), idx.end());

    for (int i = 0; i < batch_size; i++) {
        vector<float> os(batch.stacked_observations.data() + config.observation_space * idx[i], batch.stacked_observations.data() + config.observation_space * idx[i] + config.observation_space);
        vector<float> tp(batch.target_policies.data() + config.action_space * idx[i], batch.target_policies.data() + config.action_space * idx[i] + config.action_space);                   
        vector<float> tval(batch.target_values.data() + idx[i], batch.target_values.data() + idx[i] + 1);                   
        Batch sample{config, os, tp, tval};

        samples += sample;
    }
    return samples;
}
 

void Replay_buffer::append(Batch& batch_) {
    batch += batch_;
}


void Replay_buffer::save() {

}


void Replay_buffer::print_sample() {
    batch.print_sample();
}


void Replay_buffer::clear() {
    batch.clear();
}


int Replay_buffer::get_size() {
    return batch.n_samples;
}
