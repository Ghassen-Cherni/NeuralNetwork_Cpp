#include<iostream>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace Eigen;
using namespace std;



void plotVectors(const vector<float>& income, const vector<float>& prediction, const vector<float>& groundtruth) {
    // Write data to file
    std::ofstream dataFile("data.txt");
    for (size_t i = 0; i < income.size(); i++) {
        dataFile << income[i] << " " << prediction[i] << " " << groundtruth[i] << std::endl;
    }
    dataFile.close();

    // Write GNUPlot script
    std::ofstream scriptFile("script.gp");
    scriptFile << "set title 'Income vs Happiness'\n";
    scriptFile << "set xlabel 'Income'\n";
    scriptFile << "set ylabel 'Happiness'\n";
    scriptFile << "plot 'data.txt' using 1:2 with points title 'Predicted Happiness', ";
    scriptFile << "'data.txt' using 1:3 with points title 'Ground Truth Happiness', ";
    scriptFile << "'data.txt' using 1:2 with lines title 'Predicted Line'\n";
    scriptFile.close();

    // Call GNUPlot
    system("gnuplot -persist script.gp");
}

vector <VectorXf> read_csv(string filename){

    ifstream inputFile(filename);
            string line;
            getline( inputFile, line );
            vector <VectorXf> inputs; 
            while (getline( inputFile, line)){
                stringstream lineStream(line);
                string cell;
                vector<float> temp_input;
                getline(lineStream, cell, ','); //skip first column

                while (getline(lineStream, cell, ',')) {
                    temp_input.push_back(stof(cell));
                }
                inputs.push_back(Map<VectorXf> (temp_input.data(), temp_input.size()));
            }
    return inputs;
}


float loss_function(VectorXf groundTruth, VectorXf output){
            float loss = ((output - groundTruth)*((output - groundTruth).transpose())).value();
            return loss;
        }


class NeuralNetwork{
    public:
        NeuralNetwork(const vector<int>& topology, const int& input_size, const float& learning_rate){
            nb_layers = topology.size() ; //topology ex [3, 3, 1] -> 1 hidden layer
            this -> learning_rate = learning_rate;
            for (int i = 0; i < nb_layers-1; i++){
                weights.push_back(MatrixXf::Random(topology[i+1], topology[i])); // +1 to account for bias
                biases.push_back(VectorXf::Random(topology[i+1]));
                }
            };

        VectorXf activation_relu(const VectorXf& input){
            VectorXf output = input.array().max(0);    //ReLu
            return output;
        }

        VectorXf activation_relu_derivative(const VectorXf& input){
            VectorXf output;
            output = (input.array() > 0).select(1, input);
            output = (output.array() < 0).select(0, output);
            return output;
        }

        VectorXf feedForward(VectorXf input, const VectorXf& ground_truth){  //input 1*n, weights n*l 
            linear_output.clear();
            layer_outputs.clear();
            for (int i=0; i<nb_layers - 2; i++){
                linear_output.push_back(weights[i] * input + biases[i]);  // (l*n) * (n*1)  -> (l*1)
                layer_outputs.push_back(activation_relu(linear_output[i]));  // (1*4)
                input = layer_outputs[i];

            }
            input = layer_outputs.back() ;
            // input.conservativeResize(input.size() + 1);
            // input(input.size() - 1) = 1;             // (1 * 4)
            linear_output.push_back(weights.back() * input + biases.back());      // (1 * 1)
            layer_outputs.push_back(linear_output.back()); //since regression, linear activation function for last layer, 1*l
            loss = loss_function(ground_truth, layer_outputs.back());
            return layer_outputs.back();   
        }

        void backpropagation(const VectorXf& ground_truth, const VectorXf& input){
            vector<VectorXf> residuals(nb_layers - 1);   // (n*1)
            residuals.back() = layer_outputs.back() - ground_truth;     // (n*1)
            vector<MatrixXf> gradients(nb_layers - 1);

            gradients.back() = residuals.back() * layer_outputs[layer_outputs.size() - 2].transpose(); // 1*4

            for (int i = nb_layers-3; i > 0; i--){
                residuals[i] = ( (weights[i+1].transpose() * residuals[i+1]).array() * activation_relu_derivative(linear_output[i]).array());
                gradients[i] = residuals[i] * linear_output[i-1].transpose();
                
            }
            
            residuals[0] = ( ( weights[1].transpose() * residuals[1]).array() * activation_relu_derivative(linear_output[0]).array() );
            gradients[0] = residuals[0] * input.transpose();
            
            for (int i = 0; i < weights.size(); i++)
                weights[i] = weights[i] - learning_rate * gradients[i];
        }

        void train(vector <VectorXf> inputs){

            
            for (auto& elem : inputs){
                const int size = elem.size();
                VectorXf input = elem.segment(0, size - 1);
                VectorXf ground_truth = elem.segment(size - 1, 1);
                feedForward(input, ground_truth );
                backpropagation(ground_truth, input);
            }
        }

    int nb_layers;
    float loss;
    float learning_rate;
    vector<MatrixXf> weights;
    vector<VectorXf> linear_output;
    vector<VectorXf> layer_outputs;
    vector<VectorXf> biases;
    };


int main()
{
    // 1 hidden layer, feedforward
    vector <VectorXf> inputs = read_csv("./income_data/income_data.csv");
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
    for (auto& elem : train){
        const int size = elem.size();
        VectorXf input = elem.segment(0, size - 1);
        VectorXf ground_truth = elem.segment(size - 1, 1);
        VectorXf output = network.feedForward(input, ground_truth );
        predictions.push_back(output.value());
        groundtruth.push_back( ground_truth.value());
        income.push_back(input.value());
        loss += loss_function(ground_truth, output);
        i++;
    }

    loss = sqrt(loss/i);
    cout << "Training loss is " << loss << endl;

    predictions.clear();
    groundtruth.clear();
    income.clear();
    loss = 0;
    
    for (auto& elem : test){
        const int size = elem.size();
        VectorXf input = elem.segment(0, size - 1);
        VectorXf ground_truth = elem.segment(size - 1, 1);
        VectorXf output = network.feedForward(input, ground_truth );
        predictions.push_back(output.value());
        groundtruth.push_back( ground_truth.value());
        income.push_back(input.value());
        loss += loss_function(ground_truth, output);
        i++;
    }


    loss = sqrt(loss/i);
    cout << "Testing loss is " << loss << endl;

    plotVectors(income, predictions, groundtruth);
    return 0;
}   