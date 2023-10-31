#include <torch/torch.h>
#include <iostream>
#include <ros/ros.h>


using namespace std;


int main() {
  // device
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  cout << "Device : " << device << endl;

  // torch intializing
  torch::Tensor a = torch::tensor({1, 2, 3}).to(device);
  torch::Tensor b = torch::randn({1, 3});
  cout << "manual init a : " << a << endl << "random init b : " << b << endl;

  // defining tensor option
  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(true);
  b = torch::randn({1, 3}, options);
  cout << "b with options: \n" << b << endl;

  // dtype
  cout << "a dtype : " << a.dtype() << endl << "b dtype : " << b.dtype() << endl;

  // changing dytpe
  a = a.to(torch::kFloat32); // have to reassign the variable
  cout << "a dtype : " << a.dtype() << endl;

  // index refer
  cout << "a[1] : " << a.index({1}) << endl;

  // change the value at specific index
  a = a.index_put_({1}, 10);
  cout << "a : " << a << endl;

  // get tensor from vector
  vector<float> c = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  torch::Tensor d = torch::from_blob(c.data(), {3, 3});
  cout << d << endl;

  // network input example
  vector<torch::Tensor> buffer;
  for(int i = 0; i < 10; i++) {
      torch::Tensor e = torch::randn({1, 1, 160, 160});
      buffer.push_back(e);
  }
  torch::Tensor f = torch::cat({buffer[0], buffer[1]}, 1);
  cout << f.sizes() << endl;

  return 0;
}
