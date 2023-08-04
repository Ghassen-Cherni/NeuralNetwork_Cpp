#include "utils.h"
#include<fstream>
#include <sstream>

using namespace std;


void plotVectors(const vector<float>& income, const vector<float>& prediction, const vector<float>& groundtruth) {
    // Write data to file
    ofstream dataFile("data.txt");
    for (size_t i = 0; i < income.size(); i++) {
        dataFile << income[i] << " " << prediction[i] << " " << groundtruth[i] << endl;
    }
    dataFile.close();

    // Write GNUPlot script
    ofstream scriptFile("script.gp");
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


vector <VectorXf> read_csv(const string& filename){

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


float loss_function(const VectorXf& groundTruth, const VectorXf& output){
            float loss = ((output - groundTruth)*((output - groundTruth).transpose())).value();
            return loss;
        }