#include <cuda_runtime.h>
#include <chrono>
#include "my_cuda_utils.h"

using namespace chrono;

__global__ void ridge_regression(double *X, double *y, double alpha, double *XTX, double *XTy, double *L,
                                 int totalSamples, int numFeatures, int numParts)
{
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int numThreads = blockDim.x;

    int partSize = totalSamples / numParts;
    int startIdx = blockId * partSize;
    int endIdx = min(startIdx + partSize, totalSamples);

    // Step 1: Compute X^T * X
    for (int row = threadId; row < numFeatures; row += numThreads)
    {
        for (int col = 0; col < numFeatures; col++)
        {
            double sum = 0.0;
            for (int k = 0; k < totalSamples; k++)
            {
                if (k >= startIdx && k < endIdx)
                    continue;
                sum += X[k * numFeatures + row] * X[k * numFeatures + col];
            }
            XTX[blockId * numFeatures * numFeatures + row * numFeatures + col] = sum;
        }
    }

    __syncthreads();

    // // Print X^T * X matrix after computation in the desired format
    // if (threadId == 0) {
    //     printf("XTX matrix (blockId = %d):\n", blockId);
    //     for (int i = 0; i < numFeatures; i++) {
    //         for (int j = 0; j < numFeatures; j++) {
    //             printf("XTX[%d, %d] = %f\n", i, j, XTX[blockId * numFeatures * numFeatures + i * numFeatures + j]);
    //         }
    //     }
    // }

    // Step 2: Add Ridge Regularization
    if (threadId < numFeatures)
    {
        int idx = blockId * numFeatures * numFeatures + threadId * numFeatures + threadId;
        XTX[idx] += alpha + 1e-6;
        // printf("XTX[%d, %d] after regularization = %f\n", threadId, threadId, XTX[blockId * numFeatures * numFeatures + threadId * numFeatures + threadId]);
    }

    __syncthreads();

    // Step 3: Compute X^T * y
    for (int idx = threadId; idx < numFeatures; idx += numThreads)
    {
        double sum = 0.0;
        for (int i = 0; i < totalSamples; i++)
        {
            if (i >= startIdx && i < endIdx)
                continue;
            sum += X[i * numFeatures + idx] * y[i];
        }
        XTy[blockId * numFeatures + idx] = sum;
        // printf("XTy[%d] = %f\n", idx, XTy[blockId * numFeatures + idx]);
    }

    __syncthreads();

    // Step 4: Cholesky Decomposition (Only one thread per column)
    for (int i = 0; i < numFeatures; i++)
    {
        __syncthreads();

        for (int j = 0; j <= i; j++)
        {
            if (threadId == j)
            {
                double sum = 0.0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[blockId * numFeatures * numFeatures + i * numFeatures + k] *
                           L[blockId * numFeatures * numFeatures + j * numFeatures + k];
                }

                if (i == j)
                {
                    double diag_value = XTX[blockId * numFeatures * numFeatures + i * numFeatures + i] - sum;
                    if (diag_value <= 0.0)
                        return; // Check for numerical instability
                    L[blockId * numFeatures * numFeatures + i * numFeatures + j] = sqrt(diag_value);
                    // printf("L[%d, %d] = %f\n", i, i, L[blockId * numFeatures * numFeatures + i * numFeatures + j]);
                }
                else
                {
                    L[blockId * numFeatures * numFeatures + i * numFeatures + j] =
                        (XTX[blockId * numFeatures * numFeatures + i * numFeatures + j] - sum) /
                        L[blockId * numFeatures * numFeatures + j * numFeatures + j];
                    // printf("L[%d, %d] = %f\n", i, j, L[blockId * numFeatures * numFeatures + i * numFeatures + j]);
                }
            }
            __syncthreads();
        }
    }
}

void k_fold_cv(const vector<vector<double>> &X, const vector<double> &y, double &best_alpha,
               double &best_rmse, vector<double> &best_beta, int k = 5)
{
    size_t n_samples = X.size(), n_features = X[0].size() + 1;
    if (n_samples == 0 || n_features == 0 || n_samples != y.size())
        return;

    best_rmse = numeric_limits<double>::infinity();
    double alpha = 0.02, step_size = 0.01;

    vector<vector<double>> X_bias = add_intercept(X);
    vector<double> h_X = flatten(X_bias);
    vector<double> h_y = y;
    vector<double> h_XTy(k * n_features, 0.0);
    vector<double> h_L(k * n_features * n_features, 0.0);

    double *d_X, *d_y, *d_XTX, *d_XTy, *d_L;

    size_t size_X = n_samples * n_features * sizeof(double);
    size_t size_y = n_samples * sizeof(double);
    size_t size_XTX = k * n_features * n_features * sizeof(double);
    size_t size_XTy = k * n_features * sizeof(double);
    size_t size_L = k * n_features * n_features * sizeof(double);

    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_y, size_y);
    cudaMalloc(&d_XTX, size_XTX);
    cudaMalloc(&d_XTy, size_XTy);
    cudaMalloc(&d_L, size_L);

    cudaMemcpy(d_X, h_X.data(), size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), size_y, cudaMemcpyHostToDevice);

    double prev_rmse = std::numeric_limits<double>::infinity(); // Track previous RMSE
    while (alpha <= 0.02)
    {
        cudaMemset(d_XTX, 0, size_XTX);
        cudaMemset(d_XTy, 0, size_XTy);
        cudaMemset(d_L, 0, size_L);

        int numBlocks = k;
        int numThreads = 1024;

        // Launch Ridge Regression Kernel
        ridge_regression<<<numBlocks, numThreads>>>(d_X, d_y, alpha, d_XTX, d_XTy, d_L, n_samples, n_features, k);
        cudaDeviceSynchronize();

        // Copy results back to host
        cudaMemcpy(h_XTy.data(), d_XTy, size_XTy, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_L.data(), d_L, size_L, cudaMemcpyDeviceToHost);

        double avg_rmse = 0.0;
        vector<double> current_best_beta(n_features, 0.0);

        for (int fold = 0; fold < k; fold++)
        {
            // cout << "======== Fold " << fold << " ========" << endl;

            // // Print XTy values
            // cout << "XTy vector:" << endl;
            // for (int j = 0; j < n_features; j++)
            // {
            //     cout << h_XTy[fold * n_features + j] << " ";
            // }
            // cout << endl
            //      << endl;

            // // Print L matrix
            // cout << "L Matrix:" << endl;
            // for (int row = 0; row < n_features; row++)
            // {
            //     for (int col = 0; col < n_features; col++)
            //     {
            //         cout << h_L[fold * n_features * n_features + row * n_features + col] << " ";
            //     }
            //     cout << endl;
            // }
            // cout << endl;

            vector<double> beta(n_features, 0.0);

            vector<double> L_matrix(h_L.begin() + fold * n_features * n_features,
                                    h_L.begin() + (fold + 1) * n_features * n_features);

            vector<double> XTy_vector(h_XTy.begin() + fold * n_features,
                                      h_XTy.begin() + (fold + 1) * n_features);

            // Solve using Forward & Backward Substitution
            vector<double> y_temp = forward_substitution(L_matrix, XTy_vector);
            beta = backward_substitution(L_matrix, y_temp);

            // // Print y_temp vector
            // cout << "\ny_temp vector:\n";
            // for (double yt : y_temp)
            // {
            //     cout << yt << " ";
            // }
            // cout << endl;

            // Print Beta values
            cout << "\nBeta coefficients:\n";
            for (double b : beta)
            {
                cout << b << " ";
            }
            cout << endl;

            // Compute RMSE
            double fold_rmse = compute_rmse(X, y, beta, fold, k);
            avg_rmse += fold_rmse;

            // // Print RMSE for the fold
            // cout << "\nFold RMSE: " << fold_rmse << " avg " << avg_rmse << '\n'
            //      << endl;

            // Store the beta coefficients for the best fold
            current_best_beta = beta;
        }

        avg_rmse /= k; // Compute average RMSE across folds

        cout << "Alpha: " << alpha << " | RMSE: " << avg_rmse << endl;

        // Stop if RMSE starts increasing
        if (avg_rmse > prev_rmse)
        {
            cout << "Stopping early: RMSE increased at alpha = " << alpha << endl;
            break;
        }

        // Update best values if RMSE improved
        if (avg_rmse < best_rmse)
        {
            best_rmse = avg_rmse;
            best_alpha = alpha;
            best_beta = current_best_beta;
        }

        prev_rmse = avg_rmse; // Update previous RMSE
        alpha += step_size;   // Increment alpha
    }

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_XTX);
    cudaFree(d_XTy);
    cudaFree(d_L);
}

int main()
{
    vector<vector<double>> X;
    vector<double> y;
    int num_samples = 4, num_features = 4;
    int k_flods = 2;

    double best_alpha;
    double best_rmse;
    vector<double> best_beta;

    // Load data from CSV
    loadCSV("CalCOFI_processed_data.csv", X, y, num_samples, num_features);

    auto start_time = high_resolution_clock::now();

    // Run k-fold cross-validation to find best alpha and beta
    k_fold_cv(X, y, best_alpha, best_rmse, best_beta, k_flods);

    auto end_time = high_resolution_clock::now();
    double execution_time = duration<double>(end_time - start_time).count();

    // Print Results
    cout << "Total Execution Time: " << execution_time << " s" << endl;

    // Compute R² Score
    double r2_score = compute_r2_score(X, y, best_beta);

    // Print results
    cout << "Best Alpha: " << best_alpha << endl;
    cout << "Best RMSE: " << best_rmse << endl;
    cout << "R² Score: " << r2_score << endl;

    cout << "Best Ridge Coefficients (Cholesky): ";
    for (double b : best_beta)
    {
        cout << b << " ";
    }
    cout << endl;

    return 0;
}
