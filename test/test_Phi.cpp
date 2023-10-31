#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace std;

double Phi2(double param) {
    return param + param / abs(param) * 0.8;
}

double inv_Phi2(double v) {
    if(v >= 0) return max(v - 0.8, 0.0);
    else return min(v + 0.8, 0.0);
}

int main() {
    vector<float> v1;
    vector<float> v2;

    for(float i = -3.0; i <= 3.0; i += 0.1) {
        v1.push_back(Phi2(i));
        v2.push_back(inv_Phi2(i));
    }
    for(int i = 0; i < v1.size(); i++) {
        cout << v1[i] << ", ";
    }
    cout << "\n";

    for(int i = 0; i < v2.size(); i++) {
        cout << v2[i] << ", ";
    }
    cout << "\n";
}