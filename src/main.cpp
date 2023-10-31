#include <torch/torch.h>
#include <iostream>
#include <ros/ros.h>
#include <boost/thread/thread.hpp>

#include "../include/net.h"
#include "../include/data_types.h"
#include "../include/replay_buffer.h"
#include "../include/train.h"
#include "../include/mcts.h"
#include "../include/config.h"
#include "../include/utils.h"


using namespace std;

// Debug
bool DEBUG_main = false;
bool DEBUG_train = false;

// ROS Nodes
ros::Publisher Pub_StartPose, Pub_GoalPose, Pub_CollisionCheck, Pub_MctsPath, Pub_RestartScenario;
ros::Subscriber Sub_OccupancyGridMap, Sub_GoalPosition, Sub_LocalizationData, Sub_GearData, Sub_SteerData, Sub_Restart, Sub_ObstaclePoses, Sub_TerminalCase, Sub_ObstacleInfo;
geometry_msgs::PoseStamped poseStamped;
geometry_msgs::PoseArray collision_edge;
std_msgs::Header m_header;
nav_msgs::Path mcts_path;

// State Variables
vector<Coordinate> obstacle_poses;
vector<float> obstacle_info;
Coordinate goal;
Coordinate car;
State state;

// Baskets for buffer
vector<float> observation;
vector<float> observations;
vector<float> pi_s;
vector<float> z_s;

// Batch batch;

float terminal_reward = 0.0;


// Util Variables
int collision_occur = 0;
int goal_in = 0;
int time_out = 0;
int freeze_the_controller = 1;

double dummy;


// Visualize parameters
double m_collision_chk_len = 1.40;
double m_safe_width = 1.16;
double m_point_safe_region1 = 0.27;
double m_point_safe_region2 = 0.27;
double m_side_safe_region = 0.27;


template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in) {

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}


void collision_range(double x, double y, double theta) { // vehicle collision check range
    collision_edge.header = m_header;        
    std::vector<double> angle_set = linspace(0.0, 2.0 * M_PI, 20);
    geometry_msgs::PoseStamped _poseStamped;
    // Large Circle
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + m_safe_width * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + m_safe_width * sin(angle_set[i]);    //y_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    // Large Circle1
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + m_collision_chk_len*cos(theta) + m_safe_width * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + m_collision_chk_len*sin(theta) + m_safe_width * sin(angle_set[i]);    //y_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    // Large Circle2
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + 2.0 * m_collision_chk_len*cos(theta) + m_safe_width * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + 2.0 * m_collision_chk_len*sin(theta) + m_safe_width * sin(angle_set[i]);    //y_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }        

    // point circle
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x - 0.719*cos(theta) - 0.9175*sin(theta) + m_point_safe_region1 * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y - 0.719*sin(theta) + 0.9175*cos(theta) + m_point_safe_region1 * sin(angle_set[i]);    //y_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x - 0.719*cos(theta) + 0.9175*sin(theta) + m_point_safe_region1 * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y - 0.719*sin(theta) - 0.9175*cos(theta) + m_point_safe_region1 * sin(angle_set[i]);    //y_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + 2*m_collision_chk_len*cos(theta) + 0.735*cos(theta) - 0.9175*sin(theta) + m_point_safe_region2 * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + 2*m_collision_chk_len*sin(theta) + 0.735*sin(theta) + 0.9175*cos(theta) + m_point_safe_region2 * sin(angle_set[i]);    //x_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + 2*m_collision_chk_len*cos(theta) + 0.735*cos(theta) + 0.9175*sin(theta) + m_point_safe_region2 * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + 2*m_collision_chk_len*sin(theta) + 0.735*sin(theta) - 0.9175*cos(theta) + m_point_safe_region2 * sin(angle_set[i]);    //x_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }

    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + m_collision_chk_len*cos(theta) - 0.6593*cos(theta) - 0.8587*sin(theta) + m_side_safe_region * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + m_collision_chk_len*sin(theta) - 0.6593*sin(theta) + 0.8587*cos(theta) + m_side_safe_region * sin(angle_set[i]);    //x_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + m_collision_chk_len*cos(theta) - 0.6593*cos(theta) + 0.8587*sin(theta) + m_side_safe_region * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + m_collision_chk_len*sin(theta) - 0.6593*sin(theta) - 0.8587*cos(theta) + m_side_safe_region * sin(angle_set[i]);    //x_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + m_collision_chk_len*cos(theta) + 0.6593*cos(theta) - 0.8587*sin(theta) + m_side_safe_region * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + m_collision_chk_len*sin(theta) + 0.6593*sin(theta) + 0.8587*cos(theta) + m_side_safe_region * sin(angle_set[i]);    //x_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    for(int i = 0; i < angle_set.size(); i++) {
        _poseStamped.pose.position.x = x + m_collision_chk_len*cos(theta) + 0.6593*cos(theta) + 0.8587*sin(theta) + m_side_safe_region * cos(angle_set[i]);    //x_g;
        _poseStamped.pose.position.y = y + m_collision_chk_len*sin(theta) + 0.6593*sin(theta) - 0.8587*cos(theta) + m_side_safe_region * sin(angle_set[i]);    //x_g;
        _poseStamped.pose.position.z = 0.5;
        geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(0.0);
        _poseStamped.header = m_header;
        _poseStamped.pose.orientation = odom_quat;
        collision_edge.poses.push_back(_poseStamped.pose);
    }
    // if (VISUALIZATION)
    Pub_CollisionCheck.publish(collision_edge);
    collision_edge.poses.clear();
}


void Callback_Goal(const geometry_msgs::PoseArray::ConstPtr& end) { //(const geometry_msgs::PoseStamped::ConstPtr& end) 
    // cout << "Global Goal Coordinate" << endl;
    goal.th = tf::getYaw(end->poses[0].orientation);
    goal.x = end->poses[0].position.x - 1.2865250458893225 * cos(goal.th);//; + map_range/2.0;
    goal.y = end->poses[0].position.y - 1.2865250458893225 * sin(goal.th);//+ map_range/2.0;

    m_header.stamp = ros::Time::now();
    m_header.frame_id = "map";
    poseStamped.pose.position.x = goal.x;
    poseStamped.pose.position.y = goal.y;
    poseStamped.pose.position.z = 0.0;
    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(goal.th);
    poseStamped.header = m_header;
    poseStamped.pose.orientation = odom_quat;
    Pub_GoalPose.publish(poseStamped);

    poseStamped.pose.position.x = -1.2865250458893225;
    poseStamped.pose.position.y = 0.0;
    poseStamped.pose.position.z = 0.0;
    odom_quat = tf::createQuaternionMsgFromYaw(0.0);
    poseStamped.pose.orientation = odom_quat;
    Pub_StartPose.publish(poseStamped);
}


void Callback_LocalizationData(const std_msgs::Float32MultiArray::ConstPtr& msg)  {
    car.x = msg->data.at(0) - 1.2865250458893225 * cos(msg->data.at(2));
    car.y = msg->data.at(1) - 1.2865250458893225 * sin(msg->data.at(2));
    car.th = msg->data.at(2);
    // state.v = abs(msg->data.at(3));
    collision_range(car.x, car.y, car.th);
}


void Callback_ObstaclePoses(const geometry_msgs::PoseArray::ConstPtr& msg) {
    // cout << ""; // ?????
    // obstacle_poses.clear();
    // obstacle_poses.reserve(20);
    vector<Coordinate> temp;
    for(int i = 0; i < msg->poses.size(); i++) {
        double x = msg->poses[i].position.x;
        double y = msg->poses[i].position.y;
        double th = tf::getYaw(msg->poses[i].orientation);
        
        Coordinate obs_pose{x, y, th};

        temp.push_back(obs_pose);
    }
    obstacle_poses.assign(temp.begin(), temp.end());
    // cout << obstacle_poses.size() << endl;
}


void Callback_ObstacleInfo(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // obstacle_info.clear();
    // assert(obstacle_info.size() == 0);
    // vector<float>().swap(obstacle_info);
    // cout << msg->data.size() << "\n";
    obstacle_info.resize(100);
    assert(msg->data.size() == 100);
    for(int i = 0; i < msg->data.size(); i++) {
        // obstacle_info.push_back(msg->data[i]);
        obstacle_info[i] = msg->data[i];
    }
    assert(obstacle_info.size() == 100);
    // cout << "Obstacle info size : " << obstacle_info.size() << "\n"; 
    // obstacle_info.assign(msg->data.begin(), msg->data.end());
    // cout << obstacle_info << endl;
    // cout << obstacle_poses.size() << endl;
}


void Callback_TerminalCase(const std_msgs::Int32::ConstPtr& msg) {
    if(msg->data == 1) {
        goal_in = 1;
    } else if(msg->data == 2) {
        collision_occur = 1;
    } else if(msg->data == 3) {
        time_out = 1;
    } else assert(1);
}


void message_restart_scenario() {
    std_msgs::Int32 msg;
    msg.data = int(1);
    Pub_RestartScenario.publish(msg);                   
}


void Thread_ROS() {
    int argc = 0;
    char** argv;
    ros::init(argc, argv, "ROS_node");
    ros::NodeHandle nh_;
    ros::NodeHandle priv_nh("~");


    ros::CallbackQueue q;
    ros::NodeHandle nh_q;
    nh_q.setCallbackQueue(&q);
    // Asyncspinner is used to separate callback function thread
    ros::AsyncSpinner spinner(0, &q);
    spinner.start();

    Sub_GoalPosition = nh_.subscribe("/parking_cands", 1, &Callback_Goal);
    Sub_LocalizationData = nh_.subscribe("/LocalizationData", 1, &Callback_LocalizationData);
    // Sub_ObstaclePoses = nh_.subscribe("/obs_poses", 1, &Callback_ObstaclePoses);
    Sub_TerminalCase = nh_.subscribe("/terminal_case", 1, &Callback_TerminalCase);
    Sub_ObstacleInfo = nh_.subscribe("/obs_info", 1, &Callback_ObstacleInfo);
    
    Pub_StartPose = nh_.advertise<geometry_msgs::PoseStamped>("PoseStart", 1);
    Pub_GoalPose = nh_.advertise<geometry_msgs::PoseStamped>("PoseGoal", 1);
    Pub_CollisionCheck = nh_.advertise<geometry_msgs::PoseArray>("collision_edge", 1);
    Pub_MctsPath = nh_.advertise<nav_msgs::Path>("mcts_path", 1);
    Pub_RestartScenario = nh_.advertise<std_msgs::Int32>("restart_flag", 1);

    ros::spin();
}

void Thread_Observation() {
    int argc = 0;
    char** argv;
    ros::init(argc, argv, "Observation_node");
    ros::NodeHandle nh_;
    ros::NodeHandle priv_nh("~");

    sleep(2);

    Config config = _getParam(nh_);
    bool DEBUG = config.DEBUG_Main;

    ros::Rate r(40);
    while(ros::ok()) {

        state.pos = car;
        // observation = get_observation(state, goal, obstacle_info);
        // if(observation.size() != config.observation_space) {
        //     cout << observation.size() << "\n" << observation;
        // }
        // assert(observation.size() == config.observation_space);
        if(DEBUG) {
            // vector<float> v1(observation.begin(), observation.begin() + 5);
            // vector<float> v2(observation.begin() + 6, observation.end());
            // cout << "Observation\n"
            // << "Car state : [8m, 8m, pi, 540deg, 5km/h]\n" << v1 << "\n"
            // << "Obstacle Info : set of [8m, 8m, pi, 1m, 2m]\n" << v2 << "\n";
        }

        ros::spinOnce();
        r.sleep();
    }
}



void Thread_Main() {
    int argc = 0;
    char** argv;
    ros::init(argc, argv, "Main_node");
    ros::NodeHandle nh_;
    ros::NodeHandle priv_nh("~");

    cout << "Start Main Thread\n";
    Config config = _getParam(nh_);
    bool DEBUG = config.DEBUG_Main;
    
    Replay_buffer replay_buffer(config);
    int train_count = 0;
    int save_count = 0;
    int global_count = 0;
    int backup_count;
    bool stuck = false;
    vector<float> pi;
    nav_msgs::Path mcts_path;
    state.s = 0;
    state.v = 0;
    sleep(2);
    message_restart_scenario();
    sleep(2);

    torch::TensorOptions options;
    if(config.enable_gpu) {
        options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    } else {
        options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    }

    PVNet net(config);
    if(config.load_model) {
        torch::load(net, config.load_dir);
        // get global count
        cout << "Model Loaded\n";
    }

    ros::Rate r(20);
    while(ros::ok()) {
        observation = get_observation(state, goal, obstacle_info);
        // if(observation.size() != config.observation_space) {
        //     cout << observation.size() << "\n" << observation;
        // }
        assert(observation.size() == config.observation_space);
        
        
        MCTS mcts_agent(config, nh_, state, goal, obstacle_info, net, options);
        tie(state.s, state.v, pi, mcts_path, backup_count) = mcts_agent.main();
        if(global_count % 20 == 0) {
            cout << "MCTS Result - steer : " << state.s * RAD2DEG << ", velocity : " << state.v * MPS2KMPH << ", backups : " << backup_count << "\n";
        }
        if(DEBUG) {
            cout << "MCTS Policy output (pi) : " << pi << "\n";
        }
        
        Pub_MctsPath.publish(mcts_path);
        observations.insert(observations.end(), observation.begin(), observation.end());
        pi_s.insert(pi_s.end(), pi.begin(), pi.end());
        float reward = get_reward(config, state, goal);
        z_s.push_back(reward);
        if(DEBUG) {
            cout << "Immediate reward : " << reward << "\n";
        }

        if(DIST(state.pos.x, state.pos.y, goal.x, goal.y) <= 0.5 && remainder((state.pos.th - goal.th), 2 * M_PI) < 0.5 * M_PI) {
            goal_in = 1;
        }
        if(DIST(state.pos.x, state.pos.y, goal.x, goal.y) >= 15.0) {
            collision_occur = 1;
        }

        if(collision_occur || goal_in || time_out) {

            message_restart_scenario();
            
            int type = 0;
            if(collision_occur) {
                terminal_reward = -0.4;
                // freeze_the_controller = 1;            
                collision_occur = 0;
                type = 1;
            } else if(goal_in) {
                terminal_reward = 0.05 - 0.05 / 0.5 * (DIST(state.pos.x, state.pos.y, goal.x, goal.y) - 0.5);
                // freeze_the_controller = 1;
                goal_in = 0;
                type = 2;
            } else if(time_out) {
                if(abs(state.v) < 0.7) stuck = true;
                terminal_reward = -0.4;
                // freeze_the_controller = 1;
                time_out = 0;
                type = 3;
            } else assert(type == 0 && "Bad Termination");
            cout << "Scenario terminated, type " << type << "\n";

            vector<float> z_s_temp(z_s.size());
            if(DEBUG) {
                copy(z_s.begin(), z_s.end(), z_s_temp.begin());
            }
            for(int i = z_s.size() - 1; i >= 0; i--) {
                z_s[i] += terminal_reward;
                if (!stuck) {
                    terminal_reward *= config.terminal_reward_decay;
                }
            }
            stuck = false;
            if(DEBUG) {
                cout << "Compare immediate reward and value\n";
                for(int j = 0; j < z_s.size(); j += 20) {
                    cout << "element[" << j << "] : " << z_s_temp[j] << ", " << z_s[j] << "\n";
                }
            }

            Batch batch{config, observations, pi_s, z_s};
            train_count += batch.n_samples;
            save_count += batch.n_samples;
            global_count += batch.n_samples;
            if(DEBUG) {
                cout << "Current Scenario batch size : " << batch.n_samples << "\n";
            }
            
            replay_buffer.append(batch);
            if(DEBUG) {
                replay_buffer.print_sample();
            }
            cout << "Buffer size : " << replay_buffer.get_size() << "\n";

            if((int)(train_count / config.train_interval) >= 1) {
                Batch batch_sample = replay_buffer.sample(config.batch_size);
                cout << "Train data sampled\n";
                Loss loss = train(config, net, batch_sample, options);
                cout << "Train finished\n";
                train_count -= config.train_interval;
            }

            if((int)(save_count / config.save_interval) >= 1) {
                torch::save(net, config.save_dir);
                cout << "Save finished\n";
                save_count -= config.save_interval;
            }

            cout << "Reset scenario\n";
            state.v = 0;
            state.s = 0;
            observations.clear();
            pi_s.clear();
            z_s.clear();
            z_s_temp.clear();
            cout << "Observation capacity : " << observations.capacity() << "\n";
            sleep(3);
            collision_occur = 0;
            goal_in = 0;
            time_out = 0;

        }
        global_count++;

        ros::spinOnce();
        r.sleep();
    }
}


int main(int argc, char* argv[])
{
    try {
        boost::thread t1 = boost::thread(boost::bind(&Thread_ROS));
        boost::thread t2 = boost::thread(boost::bind(&Thread_Observation));
        boost::thread t3 = boost::thread(boost::bind(&Thread_Main));
        t1.join();
        t2.join();
        t3.join();
    }
    catch(std::exception& e) {
        std::cerr << "------------ error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "------------ Exception of unknown type!\n";
    }

    return 0;
} 