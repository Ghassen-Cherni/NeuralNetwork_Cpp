#include "neural_network.h"
#include "utils.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

int main()
{
    // 1 hidden layer, feedforward
    vector<VectorXf> inputs = read_csv("./income_data/income_data.csv");
    int n = static_cast<int>(inputs.size() * 0.7);
    vector<VectorXf> train(inputs.begin(), inputs.begin() + n);
    vector<VectorXf> test(inputs.begin() + n, inputs.end());

    vector<int> topology {1, 3, 1};
    NeuralNetwork network(topology, topology[0], 0.01);
    network.train(train);
    vector<float> predictions;
    vector<float> groundtruth;
    vector<float> income;
    float loss = 0;
    int i = 0;
    for (auto& elem : train) {
        const int size = elem.size();
        VectorXf input = elem.segment(0, size - 1);
        VectorXf ground_truth = elem.segment(size - 1, 1);
        VectorXf output = network.feedForward(input, ground_truth);
        predictions.push_back(output.value());
        groundtruth.push_back(ground_truth.value());
        income.push_back(input.value());
        loss += loss_function(ground_truth, output);
        i++;
    }

    loss = sqrt(loss / i);
    cout << "Training loss is " << loss << endl;

    predictions.clear();
    groundtruth.clear();
    income.clear();
    loss = 0;

    for (auto& elem : test) {
        const int size = elem.size();
        VectorXf input = elem.segment(0, size - 1);
        VectorXf ground_truth = elem.segment(size - 1, 1);
        VectorXf output = network.feedForward(input, ground_truth);
        predictions.push_back(output.value());
        groundtruth.push_back(ground_truth.value());
        income.push_back(input.value());
        loss += loss_function(ground_truth, output);
        i++;
    }

    loss = sqrt(loss / i);
    cout << "Testing loss is " << loss << endl;

    plotVectors(income, predictions, groundtruth);
    return 0;
}
