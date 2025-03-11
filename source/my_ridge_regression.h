#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include "/usr/include/eigen3/Eigen/Dense"
#include <limits>
#include <chrono>
#include <fstream>
#include <sstream>

using namespace std;
using namespace Eigen;
using namespace chrono;

// **Root Mean Square Error (RMSE)**
double rmse(const VectorXd &y_true, const VectorXd &y_pred) {
    return sqrt((y_true - y_pred).array().square().mean());
}

// **R² Calculation**
double r2_score(const VectorXd &y_true, const VectorXd &y_pred) {
    double ss_total = (y_true.array() - y_true.mean()).square().sum();
    double ss_residual = (y_true - y_pred).array().square().sum();
    return 1.0 - (ss_residual / ss_total);
}

// **Ridge Regression Using Cholesky Decomposition (with Bias)**
VectorXd ridge_cholesky_eigen(const MatrixXd &X_bias, const VectorXd &y, double alpha) {
    if (X_bias.rows() != y.size()) {
        cerr << "Error: Dimension mismatch in Ridge Regression!" << endl;
        return VectorXd();
    }

    Eigen::Index n_features = X_bias.cols();

    // Compute A = X^T * X + alpha * I
    MatrixXd A = X_bias.transpose() * X_bias + alpha * MatrixXd::Identity(n_features, n_features);
    VectorXd Xty = X_bias.transpose() * y;

    VectorXd beta = A.llt().solve(Xty);

    if (beta.rows() != n_features) {
        cerr << "Warning: Incorrect beta size! Expected " << n_features << ", got " << beta.rows() << endl;
    }

    return beta;
}

// **Load Data from CSV into Eigen Matrices**
void load_data(const string &filename, MatrixXd &X, VectorXd &y) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filename << endl;
        return;
    }

    vector<vector<double>> X_raw;
    vector<double> y_raw;
    string line;

    getline(file, line); // Skip header

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        double target_value;
        int col = 0;

        while (getline(ss, cell, ',')) {
            try {
                double value = stod(cell);
                if (col < 9) {
                    row.push_back(value);
                } else if (col == 9) {
                    target_value = value;
                }
                col++;
            } catch (const exception &e) {
                cerr << "Error parsing line: " << line << " | Exception: " << e.what() << endl;
                return;
            }
        }

        if (col == 10) {
            X_raw.push_back(row);
            y_raw.push_back(target_value);
        }
    }
    file.close();

    size_t n_samples = X_raw.size(), n_features = (n_samples > 0) ? X_raw[0].size() : 0;
    if (n_samples == 0 || n_features == 0) {
        cerr << "No valid data loaded." << endl;
        return;
    }

    X.resize(n_samples, n_features);
    y.resize(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            X(i, j) = X_raw[i][j];
        }
        y(i) = y_raw[i];
    }

    cout << "Loaded " << X.rows() << " samples with " << X.cols() << " features" << endl;
}