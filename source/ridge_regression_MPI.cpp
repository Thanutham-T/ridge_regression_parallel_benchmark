#include "./my_ridge_regression.h"
#include <mpi.h>

// **k-Fold Cross Validation with MPI**
void k_fold_cv_mpi(const MatrixXd &X, const VectorXd &y, double &best_alpha, double &best_rmse, VectorXd &best_beta, int k = 5) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t n_samples = X.rows(), n_features = X.cols();
    if (n_samples == 0 || n_features == 0 || static_cast<Eigen::Index>(n_samples) != y.size()) return;

    best_rmse = numeric_limits<double>::infinity();
    double alpha = 0.0, step_size = 0.01, prev_rmse = numeric_limits<double>::infinity();
    
    bool continue_training = true;

    while (true) {
        double local_rmse = 0.0, global_rmse = 0.0;
        int local_fold_count = 0;

        for (int fold_idx = world_rank; fold_idx < k; fold_idx += world_size) {
            // auto fold_start = high_resolution_clock::now();

            size_t fold_size = n_samples / k;
            size_t val_start = fold_idx * fold_size;
            size_t val_end = min((fold_idx + 1) * fold_size, n_samples);

            MatrixXd X_train(n_samples - fold_size, n_features);
            VectorXd y_train(n_samples - fold_size);
            MatrixXd X_val(fold_size, n_features);
            VectorXd y_val(fold_size);

            int train_idx = 0, val_idx = 0;
            for (size_t j = 0; j < n_samples; ++j) {
                if (j >= val_start && j < val_end) {
                    X_val.row(val_idx) = X.row(j);
                    y_val[val_idx++] = y[j];
                } else {
                    X_train.row(train_idx) = X.row(j);
                    y_train[train_idx++] = y[j];
                }
            }

            // **Ensure bias term is included**
            MatrixXd X_train_bias(X_train.rows(), X_train.cols() + 1);
            X_train_bias << VectorXd::Ones(X_train.rows()), X_train;
            MatrixXd X_val_bias(X_val.rows(), X_val.cols() + 1);
            X_val_bias << VectorXd::Ones(X_val.rows()), X_val;

            // **Train Ridge Regression**
            VectorXd beta = ridge_cholesky_eigen(X_train_bias, y_train, alpha);
            if (beta.size() == 0) continue;

            // **Predict and compute RMSE**
            VectorXd y_pred = X_val_bias * beta;
            double fold_error = rmse(y_val, y_pred);

            local_rmse += fold_error;
            local_fold_count++;

            // auto fold_end = high_resolution_clock::now();
            // double fold_time = duration<double, milli>(fold_end - fold_start).count();
            // cout << "Process " << world_rank << " | Fold " << fold_idx << " | Execution Time: " << fold_time << " ms" << endl;
        }

        // **Aggregate RMSE results across all processes**
        MPI_Allreduce(&local_rmse, &global_rmse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_fold_count, &k, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (k > 0) global_rmse /= k; // **Avoid division by zero**

        // **Root process updates best alpha and decides whether to stop**
        if (world_rank == 0) {
            cout << "Alpha: " << alpha << " | Avg RMSE: " << global_rmse << endl;

            if (global_rmse > prev_rmse) {
                best_alpha = alpha - step_size;
                best_rmse = prev_rmse;
                best_beta = ridge_cholesky_eigen(X, y, best_alpha);
                continue_training = false; // **Tell all processes to stop**
            } else {
                prev_rmse = global_rmse;
                best_alpha = alpha;
            }
        }

        // **Broadcast stopping condition to all processes**
        MPI_Bcast(&continue_training, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (!continue_training) break; // **Ensure all processes exit at the same time**

        // **Broadcast updated alpha to all processes**
        alpha += step_size;
        MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MatrixXd X;
    VectorXd y;

    // **Load data in root process (rank 0)**
    if (world_rank == 0) {
        load_data("./data/CalCOFI_processed_data.csv", X, y);
    }

    // **Broadcast data dimensions to all processes**
    int n_samples = X.rows(), n_features = X.cols();
    MPI_Bcast(&n_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_features, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // **Resize X and y in non-root processes**
    if (world_rank != 0) {
        X.resize(n_samples, n_features);
        y.resize(n_samples);
    }

    // **Broadcast X and y to all processes**
    MPI_Bcast(X.data(), n_samples * n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y.data(), n_samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // **Variables for best model parameters**
    double best_alpha = 0.0, best_rmse = 0.0;
    VectorXd best_beta;

    // **Start measuring execution time for the whole process**
    auto overall_start = high_resolution_clock::now();
    k_fold_cv_mpi(X, y, best_alpha, best_rmse, best_beta);
    if (world_rank == 0) {
        auto overall_end = high_resolution_clock::now();
        cout << "Total Execution Time: " << duration<double, milli>(overall_end - overall_start).count() << " ms" << endl;    
    }

    // **Only the root process computes final predictions**
    if (world_rank == 0) {
        MatrixXd X_bias(X.rows(), X.cols() + 1);
        X_bias << VectorXd::Ones(X.rows()), X;

        best_beta = ridge_cholesky_eigen(X_bias, y, best_alpha);

        VectorXd y_pred = X_bias * best_beta;

        // **Calculate R² Score**
        double r2 = r2_score(y, y_pred);

        cout << "Best Alpha: " << best_alpha << endl;
        cout << "Best RMSE: " << best_rmse << endl;
        cout << "Best R²: " << r2 << endl;
        cout << "Best Ridge Coefficients (Cholesky):\n" << best_beta.transpose() << endl;
    }

    MPI_Finalize();
    return 0;
}