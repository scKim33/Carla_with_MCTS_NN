#ifndef REPLAY_BUFFER_H_
#define REPLAY_BUFFER_H_

#include <random>

#include "data_types.h"
#include "config.h"

using namespace std;

class Replay_buffer{
public:
    Replay_buffer(Config config_);
    ~Replay_buffer();

    void append(Batch& batch_);
    Batch sample(int batch_size);
    void save();
    void print_sample();
    void clear();
    int get_size();

    
private:
    Config config;
    Batch batch;

};

#endif