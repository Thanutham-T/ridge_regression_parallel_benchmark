#include "my_ridge_regression.h"

// **k-Fold Cross Validation**
void k_fold_cv(const MatrixXd &X, const VectorXd &y, double &best_alpha, double &best_rmse, VectorXd &best_beta, int k = 5) {
    size_t n_samples = X.rows(), n_features = X.cols();
    if (n_samples == 0 || n_features == 0 || static_cast<Eigen::Index>(n_samples) != y.size()) return;

    best_rmse = numeric_limits<double>::infinity();
    double alpha = 0.0, step_size = 0.01, prev_rmse = numeric_limits<double>::infinity();

    while (true) {
        double avg_rmse = 0.0;

        for (int i = 0; i < k; ++i) {
            size_t fold_size = n_samples / k;
            size_t val_start = i * fold_size;
            size_t val_end = min((i + 1) * fold_size, n_samples);

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
            if (beta.size() == 0) return;

            // **Predict and compute RMSE**
            VectorXd y_pred = X_val_bias * beta;
            double fold_error = rmse(y_val, y_pred);

            avg_rmse += fold_error;
        }

        avg_rmse /= k;
        cout << "Alpha: " << alpha << " | Avg RMSE: " << avg_rmse << endl;

        if (avg_rmse > prev_rmse) {
            best_alpha = alpha - step_size;
            best_rmse = prev_rmse;
            best_beta = ridge_cholesky_eigen(X, y, best_alpha);
            break;
        }

        best_alpha = alpha;
        best_rmse = avg_rmse;
        prev_rmse = avg_rmse;
        alpha += step_size;
    }
}

// **Main Function**
int main() {
    MatrixXd X;
    VectorXd y;
    load_data("./data/CalCOFI_processed_data.csv", X, y);

    if (X.rows() == 0 || y.size() == 0) {
        cout << "No data loaded, exiting." << endl;
        return 1;
    }

    double best_alpha = 0.0, best_rmse = 0.0;
    VectorXd best_beta;

    auto overall_start = high_resolution_clock::now();
    k_fold_cv(X, y, best_alpha, best_rmse, best_beta);
    auto overall_end = high_resolution_clock::now();
    cout << "Total Execution Time: " << duration<double, milli>(overall_end - overall_start).count() << " ms" << endl;

    VectorXd y_pred = X * best_beta;

    // **Calculate R² Score**
    double r2 = r2_score(y, y_pred);

    // **Summary output**
    cout << "Best Alpha: " << best_alpha << endl;
    cout << "Best RMSE: " << best_rmse << endl;
    cout << "Best R²: " << r2 << endl;
    cout << "Best Ridge Coefficients (Cholesky):\n" << best_beta.transpose() << endl;

    return 0;
}