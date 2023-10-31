#ifndef UTILS_H_
#define UTILS_H_


// ROS Headers
#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>


// Headers
#include <math.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <time.h>
#include <vector>
#include <stdlib.h>
#include <torch/torch.h>


#include "data_types.h"


using namespace std;


Config _getParam(ros::NodeHandle nh_);

vector<float> get_observation(State s, Coordinate g, vector<float> obs);

float get_reward(Config config, State s, Coordinate g);

double Huber_loss(double x);

double func(double x);

double Phi(double param);

double inv_Phi(double v);

tuple<double, double, double> localization(Coordinate car, Coordinate goal);

#endif