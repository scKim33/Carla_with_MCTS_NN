#ifndef TRAIN_H_
#define TRAIN_H_

#include "data_types.h"
#include "net.h"
#include "data_types.h"
#include "config.h"

#include<torch/torch.h>

Loss train(Config config, PVNet& net, Batch& batch, torch::TensorOptions tensor_options);

#endif