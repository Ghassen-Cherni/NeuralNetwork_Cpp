#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& topology, const int& input_size, const float& learning_rate);
    VectorXf activation_relu(const VectorXf& input);
    VectorXf activation_relu_derivative(const VectorXf& input);
    VectorXf feedForward(VectorXf input, const VectorXf& ground_truth);
    void backpropagation(const VectorXf& ground_truth, const VectorXf& input);
    void train(std::vector<VectorXf> inputs);

private:
    int nb_layers;
    float loss;
    float learning_rate;
    std::vector<MatrixXf> weights;
    std::vector<VectorXf> linear_output;
    std::vector<VectorXf> layer_outputs;
    std::vector<VectorXf> biases;
};