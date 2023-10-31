#ifndef MCTS_H_
#define MCTS_H_


#include "net.h"
#include "data_types.h"
#include "config.h"
#include "utils.h"


using namespace std;

// Node Implementation
struct NODE
{
    NODE* parent; // node of the last time step
    vector<NODE*> children; // node of the next time step

    State current_state;
    
    int node_idx; // for convenience of debugging
    int n; // visit count, update at Backup stage
    double q; // expected cumulated reward, update at Backup stage
    double p; // prior, calculated from pnet
    bool is_root; // true for root node
    bool is_leaf; // true for leaf node
    int depth; // for constraining the maximum depth
    tuple<int, int> last_action_idx;

    NODE() {
        children.reserve(9);
    } // constructor

    friend ostream& operator<<(ostream &o, const NODE &n) {
        o << "Node Info" << endl
        << "Node index : " << n.node_idx << endl
        << "Parent node index : " << n.parent->node_idx << endl
        << "State " << n.current_state << endl
        << "Visit Count : " << n.n << endl
        << "Prior : " << n.p << endl
        << "Value : " << n.q << endl
        << "Root/Leaf : " << n.is_root << " / " << n.is_leaf << endl
        << "Node Depth : " << n.depth << endl
        << "Action Taken : (" << get<0>(n.last_action_idx) << ", " << get<1>(n.last_action_idx) << ")" << endl;
        return o;
    }
};


class MCTS
{
    public: 
        MCTS(Config config_, ros::NodeHandle node, State &state_, Coordinate &goal_, vector<float> &obstacles_, PVNet& net_, torch::TensorOptions tensor_options_);
        ~MCTS();

        tuple<double, double, vector<float>, nav_msgs::Path, int> main();

    private:
        // ROS Utils 
        ros::Publisher PubAction, PubPath;
        nav_msgs::Path mcts_path;
        geometry_msgs::PoseStamped point;

        // DEBUG Utils
        bool DEBUG = false;

        // Parameters
        const double MAX_S = 540; // deg
        const double MAX_V = 5; // km/h
        const double steering_outerwheelangle_ratio = 70 / 540.0;
        const double steering_innerwheelangle_ratio = 47.95 / 540.0;
        const double vehicle_length =  2.8325145696692897; // using wheel base of 2020 Benz
        double c1 = 1.4; // parameter of PUCT
        double c2 = 12345;
        double angle_weight = 1.0; // parameter of reward function, reward weight between angle and euclidian
        int MAX_DEPTH = 5; // maximum depth of MCTS Tree
        int MAX_ITER = 1000; // maximum backups of MCTS Tree
        double MAX_TIME = 0.04; // MCTS process terminate within max time
        double dt = 0.25; // time steps interval between nodes
        double gamma = 0.98; // n-step reward decay
        vector<double> steer_cand = {-20 * DEG2RAD, 0, 20 * DEG2RAD};
        vector<double> vel_cand = {-0.1 * KMPH2MPS, 0, 0.1 * KMPH2MPS};
        int backup_count = 0; // for counting how many tree searches are executed
        int global_idx = 0; // global index of the nodes

        // Nodes
        NODE root;
        vector<NODE> node_list;
        NODE* current_node;

        // Variables used for rviz
        Coordinate goal;

        // Node Init
        void InitNode(NODE& node, State state);

        // MCTS Process
        NODE* Selection(NODE* node);
        discrete_distribution<int> Expansion(NODE* node);
        double Evaluation(NODE* node);
        void BackUp(NODE* node, double reward);

        // Utils
        NODE Move(NODE* node, double ds, double v);
        double PUCT(NODE* node);
        NODE* print_child_info(NODE* node);
        void add_mcts_path(NODE* node);
        PVNetInput get_network_input(NODE* node);
        torch::Tensor get_prob(PVNetOutput out);

        // Network
        PVNet net{nullptr};
        torch::TensorOptions options;

        // Obstacle info
        vector<float> obstacles;

        Config config;

        float uniform_policy_weight;

};


#endif
