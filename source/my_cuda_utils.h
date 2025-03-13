#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>

using namespace std;

// Function to load CSV into GPU memory
void loadCSV(const string &filename, vector<vector<double>> &X, vector<double> &y, int &num_samples, int &num_features)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    string line;
    vector<vector<double>> data;
    bool isFirstRow = true;

    while (getline(file, line))
    {
        stringstream ss(line);
        vector<double> row;
        string cell;

        while (getline(ss, cell, ','))
        {
            try
            {
                cell.erase(0, cell.find_first_not_of(" \t\r\n")); // Trim leading spaces
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1); // Trim trailing spaces

                if (cell.empty() || cell == "NA")
                {
                    row.push_back(0.0); // Replace empty or "NA" values with 0.0
                }
                else
                {
                    row.push_back(stod(cell)); // Convert to double
                }
            }
            catch (const invalid_argument &e)
            {
                if (isFirstRow)
                {
                    row.clear();
                    break; // Skip header row
                }
                else
                {
                    cerr << "Error: Non-numeric value found -> '" << cell << "' in row " << data.size() + 1 << endl;
                    exit(1);
                }
            }
        }

        if (!row.empty())
        {
            data.push_back(row);
            isFirstRow = false; // Mark first row as processed
        }
    }

    file.close();

    if (data.empty())
    {
        cerr << "Error: No valid data found in the CSV file." << endl;
        exit(1);
    }

    num_samples = data.size();
    num_features = data[0].size() - 1; // Last column is the target (y)

    X.resize(num_samples, vector<double>(num_features));
    y.resize(num_samples);

    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            X[i][j] = data[i][j]; // Store features in 2D format
        }
        y[i] = data[i][num_features]; // Target variable
    }

    cout << "Loaded CSV: " << num_samples << " samples, " << num_features << " features" << endl;
}

vector<vector<double>> add_intercept(const vector<vector<double>> &X)
{
    vector<vector<double>> X_intercept(X.size(), vector<double>(X[0].size() + 1, 1.0));
    for (size_t i = 0; i < X.size(); ++i)
    {
        copy(X[i].begin(), X[i].end(), X_intercept[i].begin() + 1);
    }
    return X_intercept;
}

vector<double> flatten(const vector<vector<double>> &X)
{
    vector<double> flat_X;
    flat_X.reserve(X.size() * X[0].size());
    for (const auto &row : X)
    {
        copy(row.begin(), row.end(), back_inserter(flat_X));
    }
    return flat_X;
}

vector<double> forward_substitution(const vector<double> &L, const vector<double> &b)
{
    int N = b.size();
    vector<double> y(N, 0.0);
    for (int i = 0; i < N; i++)
    {
        double sum = b[i];
        for (int j = 0; j < i; j++)
        {
            sum -= L[i * N + j] * y[j];
        }
        y[i] = sum / L[i * N + i];
    }
    return y;
}

vector<double> backward_substitution(const vector<double> &L, const vector<double> &y)
{
    int N = y.size();
    vector<double> x(N, 0.0);
    for (int i = N - 1; i >= 0; --i)
    {
        double sum = 0.0;
        for (int j = i + 1; j < N; ++j)
        {
            sum += L[j * N + i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i * N + i];
    }
    return x;
}

double compute_rmse(const vector<vector<double>> &X, const vector<double> &y, const vector<double> &beta, int fold, int k)
{
    int n_samples = X.size();
    int n_features = X[0].size();  // The number of features, including intercept
    double sum_error = 0.0;
    int start_idx = fold * (n_samples / k);
    int end_idx = (fold + 1) * (n_samples / k);

    for (int i = start_idx; i < end_idx; i++)
    {
        // Prediction calculation with intercept
        double predicted = beta[0];  // Start with intercept (beta[0])

        // Loop through features and calculate weighted sum
        for (int j = 0; j < n_features; j++)  // Do not skip the intercept column here
        {
            predicted += X[i][j] * beta[j + 1];  // Add weighted features
        }

        // Compute the squared error and add it to the sum of errors
        double diff = predicted - y[i];  // Difference between predicted and actual value
        sum_error += pow(diff, 2);       // Add squared error to sum_error

        // // Optionally print the prediction, actual value, and the difference for debugging
        // cout << "Sample " << i + 1 << ": Predicted = " << predicted
        //      << " | Actual = " << y[i] << " | Diff = " << diff << endl;

        // // Optionally print beta values
        // cout << "Beta values: ";
        // for (int j = 0; j < beta.size(); j++)
        // {
        //     cout << beta[j] << " ";  // Print each coefficient in beta
        // }
        // cout << endl;
    }

    // Compute RMSE (Root Mean Squared Error)
    double rmse = sqrt(sum_error / (end_idx - start_idx));

    // // Print RMSE for the fold
    // cout << "RMSE for fold " << fold << ": " << rmse << endl;

    return rmse;
}

double compute_r2_score(const vector<vector<double>> &X, const vector<double> &y, const vector<double> &beta)
{
    double y_mean = 0.0;
    int n_samples = y.size();

    // Calculate the mean of y
    for (double yi : y)
    {
        y_mean += yi;
    }
    y_mean /= n_samples;

    double ss_total = 0.0, ss_residual = 0.0;
    for (int i = 0; i < n_samples; i++)
    {
        // Calculate prediction (y_pred) for sample i
        double y_pred = beta[0];  // Start with intercept
        for (int j = 0; j < X[i].size(); j++)  // Loop through features (j = 0 to n_features - 1)
        {
            y_pred += X[i][j] * beta[j + 1];  // Add weighted feature contributions
        }

        // Compute Total Sum of Squares (ss_total) and Residual Sum of Squares (ss_residual)
        ss_total += pow(y[i] - y_mean, 2);
        ss_residual += pow(y[i] - y_pred, 2);
    }

    // Compute R² score: 1 - (ss_residual / ss_total)
    return 1.0 - (ss_residual / ss_total);
}

double calculate_gflop(size_t n, size_t d, int k) {
    double flops_per_fold = (n * d * d) + (n * d) + (d * d * d) + (2 * n);
    double total_flops = k * flops_per_fold;
    return total_flops / 1e9;  // Convert to GFLOP
}
