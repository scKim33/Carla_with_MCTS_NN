#include <torch/torch.h>
#include <iostream>
#include <ros/ros.h>


#include "../include/data_types.h"
#include "../include/replay_buffer.h"
#include "../include/config.h"

using namespace std;


int main() {
//     // generate n batches of data and put it in replay buffer
//     int n_data = 3;
//     Replay_buffer replay_buffer;
//     vector<float> os;
//     cout << os.size() << endl;

//     for(int i = 0; i < n_data; i++) {
//         torch::Tensor t = torch::randn({10});
//         vector<float> o(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
//         os.reserve( os.size() + o.size() );
//         os.insert( os.end(), o.begin(), o.end() );
//     }
//     cout << os.size() << endl;

//     torch::Tensor s_ = torch::nn::functional::one_hot(torch::randint(5, {n_data,}), 5).to(torch::kFloat32);
//     torch::Tensor v_ = torch::nn::functional::one_hot(torch::randint(2, {n_data,}), 2).to(torch::kFloat32);
//     torch::Tensor ts_ = torch::nn::functional::one_hot(torch::randint(5, {n_data,}), 5).to(torch::kFloat32);
//     torch::Tensor tv_ = torch::nn::functional::one_hot(torch::randint(2, {n_data,}), 2).to(torch::kFloat32);
//     torch::Tensor val_ = torch::randn({n_data,});
//     torch::Tensor tval_ = torch::randn({n_data,});
//     vector<float> s(s_.data_ptr<float>(), s_.data_ptr<float>() + s_.numel());
//     vector<float> v(v_.data_ptr<float>(), v_.data_ptr<float>() + v_.numel());
//     vector<float> ts(ts_.data_ptr<float>(), ts_.data_ptr<float>() + ts_.numel());
//     vector<float> tv(tv_.data_ptr<float>(), tv_.data_ptr<float>() + tv_.numel());
//     vector<float> val(val_.data_ptr<float>(), val_.data_ptr<float>() + val_.numel());
//     vector<float> tval(tval_.data_ptr<float>(), tval_.data_ptr<float>() + tval_.numel());

//     cout << "os : " << os << endl << "s : " << s << endl << "v : " << v << endl <<
//     "ts : " << ts << endl << "tv : " << tv << endl << "val : " << val << endl << "tval : " << tval << endl;

//     Batch batch{os, s, v, val, ts, tv, tval};
//     // Batch batch;
//     replay_buffer.append(batch);
//     replay_buffer.print_sample();

//     // given buffer, vector->tensor and make training process, now, generate 1000 data in batch
//     n_data = 1000;
//     replay_buffer.clear();

//     os.clear();

//     for(int i = 0; i < n_data; i++) {
//         torch::Tensor t = torch::randn({10});
//         vector<float> o(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
//         os.reserve( os.size() + o.size() );
//         os.insert( os.end(), o.begin(), o.end() );
//     }
//     cout << os.size() << endl;

//     s_ = torch::nn::functional::one_hot(torch::randint(5, {n_data,}), 5).to(torch::kFloat32);
//     v_ = torch::nn::functional::one_hot(torch::randint(2, {n_data,}), 2).to(torch::kFloat32);
//     ts_ = torch::nn::functional::one_hot(torch::randint(5, {n_data,}), 5).to(torch::kFloat32);
//     tv_ = torch::nn::functional::one_hot(torch::randint(2, {n_data,}), 2).to(torch::kFloat32);
//     val_ = torch::randn({n_data,});
//     tval_ = torch::randn({n_data,});
//     s.assign(s_.data_ptr<float>(), s_.data_ptr<float>() + s_.numel());
//     v.assign(v_.data_ptr<float>(), v_.data_ptr<float>() + v_.numel());
//     ts.assign(ts_.data_ptr<float>(), ts_.data_ptr<float>() + ts_.numel());
//     tv.assign(tv_.data_ptr<float>(), tv_.data_ptr<float>() + tv_.numel());
//     val.assign(val_.data_ptr<float>(), val_.data_ptr<float>() + val_.numel());
//     tval.assign(tval_.data_ptr<float>(), tval_.data_ptr<float>() + tval_.numel());

//     batch = Batch({os, s, v, val, ts, tv, tval});
//     // Batch batch;
//     replay_buffer.append(batch);
//     replay_buffer.print_sample();

//     Batch sample = replay_buffer.sample(64);
//     cout << sample << endl;

    return 0;
}

