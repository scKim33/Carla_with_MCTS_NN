# Carla with Monte Carlo Tree Search(MCTS)

## Prerequisites
- heightmap package
- carla server
- roscore

## Code configuration
![code_config](https://github.com/scKim33/Carla_with_MCTS/blob/main/fig/1.png)

## Structure
```
mcts_network
│   ├── carla_simulator
│   │   ├── carla-0.9.12-py3.8-linux-x86_64_with_full_seg.egg
│   │   └── test_scenario.py
│   ├── CMakeLists.txt
│   ├── config
│   │   └── config.yaml
│   ├── dataset
│   │   ├── t10k-images-idx3-ubyte
│   │   ├── t10k-labels-idx1-ubyte
│   │   ├── train-images-idx3-ubyte
│   │   └── train-labels-idx1-ubyte
│   ├── fig
│   ├── include
│   │   ├── config.h
│   │   ├── data_types.h
│   │   ├── mcts.h
│   │   ├── net.h
│   │   ├── replay_buffer.h
│   │   ├── train.h
│   │   └── utils.h
│   ├── launch
│   │   ├── mcts_network.launch
│   │   ├── test_mcts.launch
│   │   ├── test_MNIST_train.launch
│   │   ├── test_network_train.launch
│   │   ├── test_Phi.launch
│   │   ├── test_replay_buffer.launch
│   │   └── test_tensor_basics.launch
│   ├── package.xml
│   ├── README.md
│   ├── rviz
│   │   └── rviz.rviz
│   ├── save
│   │   └── model.pt
│   ├── src
│   │   ├── main.cpp
│   │   ├── mcts.cpp
│   │   ├── net.cpp
│   │   ├── replay_buffer.cpp
│   │   ├── train.cpp
│   │   └── utils.cpp
│   └── test
│       ├── test_mcts.cpp
│       ├── test_MNIST_train.cpp
│       ├── test_network_train.cpp
│       ├── test_Phi.cpp
│       ├── test_replay_buffer.cpp
│       └── test_tensor_basics.cpp

```
- test_scenario.py : create carla actor
- config.yaml : set the parameters of mcts search
- mcts_random.cpp : mcts loop executed, called by main.cpp

## How to use
1. ROS master
```
roscore
```
2. Carla Server
```
./CarlaUE4.sh
```
3. Carla Actor
```
cd carla_simulator
python3 test_scenario.py
```
4. Run MCTS using roslaunch command
```
roslaunch mcts_network mcts_network.launch
```