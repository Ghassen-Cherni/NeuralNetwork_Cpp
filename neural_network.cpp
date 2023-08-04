#include "neural_network.h"
#include "utils.h"
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

NeuralNetwork::NeuralNetwork(const vector<int>& topology, const int& input_size, const float& learning_rate) {
    nb_layers = topology.size();
    this->learning_rate = learning_rate;
    for (int i = 0; i < nb_layers - 1; i++) {
        weights.push_back(MatrixXf::Random(topology[i + 1], topology[i]));
        biases.push_back(VectorXf::Random(topology[i + 1]));
    }
}

VectorXf NeuralNetwork::activation_relu(const VectorXf& input) {
    VectorXf output = input.array().max(0);
    return output;
}

VectorXf NeuralNetwork::activation_relu_derivative(const VectorXf& input) {
    VectorXf output;
    output = (input.array() > 0).select(1, input);
    output = (output.array() < 0).select(0, output);
    return output;
}

VectorXf NeuralNetwork::feedForward(VectorXf input, const VectorXf& ground_truth) {
    linear_output.clear();
    layer_outputs.clear();
    for (int i = 0; i < nb_layers - 2; i++) {
        linear_output.push_back(weights[i] * input + biases[i]);
        layer_outputs.push_back(activation_relu(linear_output[i]));
        input = layer_outputs[i];
    }
    input = layer_outputs.back();
    linear_output.push_back(weights.back() * input + biases.back());
    layer_outputs.push_back(linear_output.back());
    loss = loss_function(ground_truth, layer_outputs.back());
    return layer_outputs.back();
}

void NeuralNetwork::backpropagation(const VectorXf& ground_truth, const VectorXf& input) {
    vector<VectorXf> residuals(nb_layers - 1);
    residuals.back() = layer_outputs.back() - ground_truth;
    vector<MatrixXf> gradients(nb_layers - 1);

    gradients.back() = residuals.back() * layer_outputs[layer_outputs.size() - 2].transpose();

    for (int i = nb_layers - 3; i > 0; i--) {
        residuals[i] = ((weights[i + 1].transpose() * residuals[i + 1]).array() * activation_relu_derivative(linear_output[i]).array());
        gradients[i] = residuals[i] * linear_output[i - 1].transpose();
    }

    residuals[0] = ((weights[1].transpose() * residuals[1]).array() * activation_relu_derivative(linear_output[0]).array());
    gradients[0] = residuals[0] * input.transpose();

    for (int i = 0; i < weights.size(); i++)
        weights[i] = weights[i] - learning_rate * gradients[i];
}

void NeuralNetwork::train(vector<VectorXf> inputs) {
    for (auto& elem : inputs) {
        const int size = elem.size();
        VectorXf input = elem.segment(0, size - 1);
        VectorXf ground_truth = elem.segment(size - 1, 1);
        feedForward(input, ground_truth);
        backpropagation(ground_truth, input);
    }
}
