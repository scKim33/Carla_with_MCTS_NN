#include "../include/mcts.h"

using namespace std;


MCTS::MCTS(Config config_, ros::NodeHandle node, State &state_, Coordinate &goal_, vector<float> &obstacles_, PVNet& net_, torch::TensorOptions tensor_options_) {
    // Generate ROS nodes
    PubAction =  node.advertise<std_msgs::Float32MultiArray>("Control_Command", 1);


    config = config_;
    DEBUG = config.DEBUG_MCTS;
    c1 = config.c1;
    c2 = config.c2;
    angle_weight = config.angle_weight;
    MAX_DEPTH = config.MAX_DEPTH;
    MAX_ITER = config.MAX_ITER;
    MAX_TIME = config.MAX_TIME;
    dt = config.dt;
    gamma = config.gamma;
    uniform_policy_weight = config.uniform_policy_weight;
    options = tensor_options_;


    if(DEBUG) {
        cout << "Generate MCTS Agent" << endl;
        cout << "Debugging Mode" << endl;
    }

    net = net_;
    goal = goal_;
    obstacles = obstacles_;
    
    node_list.reserve(50000); // allocate enough memory of global node list
    node_list.clear();

    // Root node initialize
    InitNode(root, state_);
    root.is_root = true;
    root.depth = 0;
    node_list.push_back(root); // save node at global variable

    current_node = &node_list[0];
    node_list[0].parent = &node_list[0]; // for root node especially, take closed form
}

MCTS::~MCTS() {

}

tuple<double, double, vector<float>, nav_msgs::Path, int> MCTS::main() {
    if (DEBUG)
        cout << "Coordinate information of root node: (" << root.current_state.pos.x << ", " << root.current_state.pos.y << ", " << root.current_state.pos.th << ")" << endl;

    ros::Time time_flag = ros::Time::now();
    if (DEBUG) {
        cout << "MCTS Start" << endl;
    }
    while ((ros::Time::now() - time_flag).toSec() < MAX_TIME && backup_count < MAX_ITER) { //tree search until time limit
        if(DEBUG) {
            // cout << "\n\nThis value must be true... is root? : " << current_node->is_root << endl;
        }
        
        while(!(current_node->is_leaf)) {
            current_node = Selection(current_node);
        }
        if(DEBUG) {
            // cout << "\n\nleaf node found, idx info : " << current_node->node_idx << ", depth info : " << current_node->depth << endl;
        }

        if(current_node->depth < MAX_DEPTH) {
            discrete_distribution<int> dis = Expansion(current_node);
        }

        double r = Evaluation(current_node);

        BackUp(current_node, r);

        backup_count++;
    }

    if(DEBUG) {
        cout << "Finished Tree Search, current node should be root... is root? : " << current_node->is_root << endl;
        cout << "Root Info\n"
            << "x : " << root.current_state.pos.x - goal.x << ", y : " << root.current_state.pos.y - goal.y << ", th : " << (root.current_state.pos.th - goal.th) * RAD2DEG
            << ", s : " << root.current_state.s * RAD2DEG << ", v : " << root.current_state.v * MPS2KMPH << "\n";

        cout << "Tree Search Result\n";
    }

    NODE* best_child;
    while(!current_node->is_leaf) {
        add_mcts_path(current_node);

        current_node = print_child_info(current_node);
        if(current_node->parent->is_root) {
            best_child = current_node;
        }
    }

    tuple<int, int> last_action_idx_ = best_child->last_action_idx;
    double s = root.current_state.s + steer_cand[get<0>(last_action_idx_)];
    double v = root.current_state.v + vel_cand[get<1>(last_action_idx_)];
    if (s * RAD2DEG > MAX_S) {
        s = MAX_S * DEG2RAD;
    } else if (-MAX_S > s * RAD2DEG) {
        s = -MAX_S * DEG2RAD;
    }
    if(v * MPS2KMPH > MAX_V) {
        v = MAX_V * KMPH2MPS;
    } else if(v * MPS2KMPH < -MAX_V) {
        v = -MAX_V * KMPH2MPS;
    }
    
    std_msgs::Float32MultiArray msg;
    msg.data.push_back(s * RAD2DEG);
    msg.data.push_back(v * MPS2KMPH); // deg, km/h
    PubAction.publish(msg);

    if(DEBUG) {
        cout.precision(4);
        cout << "Control Command { Steer : " << msg.data[0] << "[deg] , Velocity : " << msg.data[1] << "[km/h] }\n"
        << "Calc time(ms) : " << (ros::Time::now() - time_flag).toSec() * 1000
        << ", # backups : " << backup_count
        << ", # nodes generated : " << global_idx << "\n\n";

    }

    vector<float> pi;
    for(int i = 0; i < config.action_space; i++) {
        pi.push_back((1 - uniform_policy_weight) * node_list[0].children[i]->n / (float)node_list[0].n
                        + uniform_policy_weight * 1 / config.action_space);
    }

    return make_tuple(s, v, pi, mcts_path, backup_count);
}


void MCTS::InitNode(NODE& node, State state) {
    node.n = 0;
    node.q = 0;
    node.is_root = false;
    node.is_leaf = true;
    node.node_idx = global_idx++;
    node.current_state = state;
}


NODE* MCTS::Selection(NODE* node) {
    if (DEBUG) {
        cout << "\n\n============ Selection ============" << endl;
    }
    
    // Iterate the children node and return the children which has the best PUCT
    int bestidx = 0;
    for(int i = 0; i < node->children.size(); i++) {
        if(DEBUG) {
            cout << "selection candidate idx : " << node->children[i]->node_idx;
            cout << ", candidate puct : " << PUCT(node->children[i]) << endl;
        }
        if (PUCT(node->children[i]) > PUCT(node->children[bestidx])) {
            bestidx = i;
        }
    }

    if (DEBUG) {
        cout << "Best node index is : " << node->children[bestidx]->node_idx << endl;
    }
    return node->children[bestidx];
}


discrete_distribution<int> MCTS::Expansion(NODE* node) {
    // Generate (action-dim) childeren nodes for given node
    // Each children nodes should be initialized
    if (DEBUG) {
        cout << "\n\n============ Expansion ============" << endl;
    }

    torch::Tensor input = get_network_input(node);

    PVNetOutput out = net->forward(input);

    node->q = out.value[0].item<float>();

    torch::Tensor policy_table_ = get_prob(out).to(torch::kCPU);
    vector<float> policy_table(policy_table_.data_ptr<float>(), policy_table_.data_ptr<float>() + policy_table_.numel());
    discrete_distribution<int> policy_table_distribution(policy_table_.data_ptr<float>(), policy_table_.data_ptr<float>() + policy_table_.numel());

    node->is_leaf = false;
    for(int j = 0; j < vel_cand.size(); j++) {
        for(int i = 0; i < steer_cand.size(); i++) {
            double ds = steer_cand[i];
            double dv = vel_cand[j];

            NODE child_node = Move(node, ds, dv); // local_variable
            child_node.parent = node;
            child_node.depth = node->depth + 1;
            child_node.last_action_idx = make_tuple(i, j);
            double l = 1.0;
            child_node.p = l * policy_table[steer_cand.size() * j + i] + 1 / (vel_cand.size() * steer_cand.size()) * (1.0 - l); // uniform

            node_list.push_back(child_node); // save node at global variable
            if(DEBUG) {
                cout << child_node << endl;
            }
            node->children.push_back(&node_list[global_idx - 1]);
        }
    }
    return policy_table_distribution;
}


double MCTS::Evaluation(NODE* node) {
    if (DEBUG) {
        cout << "\n\n========== Evaluation ==========" << endl;
        cout << "Evaluated value : " << node->q << endl;
    }

    return node->q;
}


void MCTS::BackUp(NODE* node, double reward) {
    if (DEBUG) {
        cout << "\n\n========== Backup ==========" << endl;
        cout << "node idx : " << node->node_idx << endl;
        cout << "is root node? : " << node->is_root << endl;
        cout << "current node state : " << node->current_state << endl;
        cout << "q before backup : " << node->q << endl;
        cout << "current node visit count : " << node->n << endl;
    }

    // update the node info
    double q_sum = node->q * node->n + reward;
    node->n += 1;
    node->q = q_sum / node->n;

    if(DEBUG) {
        cout << "q after backup : " << node->q << endl;
    }

    if(!(node->is_root)){
        if(DEBUG) {
            cout << "next backup node idx : " << node->parent->node_idx << endl;
        }
        current_node = node->parent;
        BackUp(current_node, gamma * reward);
    }
}


NODE MCTS::Move(NODE* node, double ds, double dv) { // give an action as input and return new node with next position
    // Assuming constant v, s while moving
    double new_s = node->current_state.s + ds;
    double param = inv_Phi(node->current_state.v) + dv;
    double new_v = Phi(param);
    assert(abs(new_v) >= 0.8 || abs(new_v) == 0);

    // Physical constraint of the car
    if(new_s * RAD2DEG > MAX_S) {
        new_s = MAX_S * DEG2RAD;
    } else if(new_s * RAD2DEG < -MAX_S) {
        new_s = -MAX_S * DEG2RAD;
    }
    if(new_v * MPS2KMPH > MAX_V) {
        new_v = MAX_V * KMPH2MPS;
    } else if(new_v * MPS2KMPH < -MAX_V) {
        new_v = -MAX_V * KMPH2MPS;
    }

    double wheel_angle = 1 / 2.0 * (steering_outerwheelangle_ratio + steering_innerwheelangle_ratio) * new_s;

    // Next point calculation with interval dt
    double dx = new_v * dt * cos(node->current_state.pos.th);
    double dy = new_v * dt * sin(node->current_state.pos.th);
    double dth = (new_v * dt / vehicle_length) * tan(-wheel_angle);

    State next_state;
    next_state.pos.x = node->current_state.pos.x + dx;
    next_state.pos.y = node->current_state.pos.y + dy;
    next_state.pos.th = remainder((node->current_state.pos.th + dth), 2 * M_PI);
    next_state.s = new_s;
    next_state.v = new_v;

    if(DEBUG) {
        cout << "Move Info" << endl;
        cout << "x = " << node->current_state.pos.x << ", y = " << node->current_state.pos.y << ", th = " << node->current_state.pos.th * RAD2DEG << ", s = " << node->current_state.s << ", v = " << node->current_state.v << endl;
        cout << "dx = " << dx << ", dy = " << dy << ", dth = " << dth * RAD2DEG << ", ds = " << ds << ", dv = " << dv << endl;
        cout << "x = " << next_state.pos.x << ", y = " << next_state.pos.y << ", th = " << next_state.pos.th * RAD2DEG << ", s = " << next_state.s << ", v = " << next_state.v << endl;
    }

    NODE next_node;
    InitNode(next_node, next_state);
    return next_node;
}


double MCTS::PUCT(NODE* node) {
    return node->q + node->p * sqrt(node->parent->n) / (1.0 + node->n) * (c1 + log((node->parent->n + c2 + 1) / c2));
}


NODE* MCTS::print_child_info(NODE* node) {
    int bestidx = 0;
    cout << "Current node depth: " << node->depth << ", Print child nodes info.\n";

    for(int i = 0; i < node->children.size(); i++) {
        cout.precision(2);
        cout << "Node index: " << node->node_idx
        << ", dv : " << setw(2) << vel_cand[get<1>(node->children[i]->last_action_idx)] * MPS2KMPH << "[km/h]"
        << ", ds : " << setw(3) << steer_cand[get<0>(node->children[i]->last_action_idx)] * RAD2DEG << "[deg]";
        cout.precision(5);
        cout << ", # Visit : " << setw(5) << node->children[i]->n
        << ", Q-value : " << setw(10) << node->children[i]->q
        << ", PUCT : " << setw(10) << PUCT(node->children[i]) << endl;

        if (node->children[i]->n > node->children[bestidx]->n) {
            bestidx = i;
        }
    }
    return node->children[bestidx];
}


void MCTS::add_mcts_path(NODE* node) {
    mcts_path.header.frame_id = point.header.frame_id = "map";
    mcts_path.header.stamp = point.header.stamp = ros::Time::now();
    point.pose.position.x = node->current_state.pos.x;
    point.pose.position.y = node->current_state.pos.y;
    point.pose.position.z = 0.5;
    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(node->current_state.pos.th);
    point.pose.orientation = odom_quat;
    mcts_path.poses.push_back(point);
}

PVNetInput MCTS::get_network_input(NODE* node) {
    double x, y, th;
    {x, y, th} = localization(node->current_state.pos, goal);
    double s = node->current_state.s;
    double v = node->current_state.v;
    torch::Tensor car_state_normalized = torch::tensor({(float)((x + 30.0) / 60.0),
                                          (float)((y + 30.0) / 60.0),
                                          (float)((th + M_PI) / (2 * M_PI)),
                                          (float)((s * RAD2DEG + 540.0) / (1080.0)),
                                          (float)((v * MPS2KMPH + 5.0) / (10.0))},
                                          options);
    vector<float> temp(100);
    for(int i = 0; i < obstacles.size(); i += 5) {
        double x_, y_, th_;
        Coordinate obs_pose({obstacles[i], obstacles[i+1], obstacles[i+2]});
        {x_, y_, th_} = localization(obs_pose, node->current_state.pos);
        temp[i] = (float)((x_ + 30.0) / 60.0);
        temp[i+1] = (float)((y_ + 30.0) / 60.0);
        temp[i+2] = (float)((th_ + M_PI) / (2 * M_PI));
        temp[i+3] = obstacles[i+3];
        temp[i+4] = obstacles[i+4];
    }
    torch::Tensor obstacle_input_normalized = torch::from_blob(temp.data(), {config.observation_space - 5}, options);
    return torch::cat({car_state_normalized, obstacle_input_normalized}); // when using gpu, cuda:0, cuda:-2 different device error
}

torch::Tensor MCTS::get_prob(PVNetOutput out) {
    return torch::softmax(out.policy, 0);
}


