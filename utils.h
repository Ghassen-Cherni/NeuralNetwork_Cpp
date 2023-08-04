#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

vector<VectorXf> read_csv(const string& filename);
float loss_function(const VectorXf&  groundTruth, const VectorXf&  output);
void plotVectors(const std::vector<float>& income, const std::vector<float>& prediction, const std::vector<float>& groundtruth);