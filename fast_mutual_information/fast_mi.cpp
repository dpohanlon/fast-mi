#include <vector>
#include <cmath>
#include <numeric>

#ifdef ENABLE_BENCHMARK
#include <benchmark/benchmark.h>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <functional>

#include <Eigen/Dense>

#include "kdtree.hpp"
#include "tests.hpp"
#include "mvn.hpp"
#include "utils.hpp"
#include "mutual_information.hpp"

// int main() {

//     Eigen::VectorXd mean(2);
//     mean << 40.0, 150.0;

//     Eigen::VectorXd variance(2);
//     variance << 30.0, 50.0;

//     Eigen::MatrixXd corr(2, 2);
//     corr <<  1.0,  -0.7,
//             -0.7,  1.0;

//     const double step = 0.01;

//     std::vector<double> rho_vec;
//     std::vector<double> mi_analytical_vec;
//     std::vector<double> mi_tree_vec;

//     for (double rho = -0.99; rho <= 1.0; rho += step) {

//         rho_vec.push_back(rho);

//         corr(0, 1) = rho;
//         corr(1, 0) = rho;

//         Eigen::MatrixXd cov = correlationToCovariance(corr, variance);

//         int num_samples = 1000000;

//         Eigen::MatrixXd samples = sampleMultivariateNormal(mean, cov, num_samples);

//         Eigen::VectorXd std_dev = variance.array().sqrt();
//         // Eigen::MatrixXd uniform_samples = transformToUniform(samples, mean, std_dev);

//         double mi = mutual_information_quantised(mean(0), std_dev(0), mean(1), std_dev(1), samples, 100);

//         double analyticalMI = -0.5 * std::log(1. - std::pow(corr(0, 1), 2.));

//         mi_analytical_vec.push_back(analyticalMI);
//         mi_tree_vec.push_back(mi);

//     }

//     std::string csv_filename = "mi_100k.csv";

//     std::ofstream file(csv_filename);

//     file << "rho,mi,mi_tree\n";

//     file << std::fixed << std::setprecision(6);

//     for (int i = 0; i < rho_vec.size(); i++) {
//         file << rho_vec[i] << "," << mi_analytical_vec[i] << "," << mi_tree_vec[i] << "\n";
//     }

//     file.close();

// }

static void BM_MI(benchmark::State& state) {

    Eigen::VectorXd mean(2);
    mean << 40.0, 150.0;

    Eigen::VectorXd variance(2);
    variance << 30.0, 50.0;

    Eigen::MatrixXd corr(2, 2);
    corr <<  1.0,  -0.7,
            -0.7,  1.0;

    Eigen::MatrixXd cov = correlationToCovariance(corr, variance);

    int num_samples = 10000;

    Eigen::MatrixXd samples = sampleMultivariateNormal(mean, cov, num_samples);

    Eigen::VectorXd std_dev = variance.array().sqrt();

    for (auto _ : state) {
        std::vector<float> results(state.range(0));
        for (size_t i = 0; i < results.size(); ++i) {

            double mi = mutual_information_quantised(mean(0), std_dev(0), mean(1), std_dev(1), samples, 10);

            results[i] = mi;
        }
        benchmark::DoNotOptimize(results);
    }
    state.SetComplexityN(state.range(0));
}

Eigen::MatrixXd sampleMultivariateNormal2(const Eigen::VectorXd &mean,
                                           const Eigen::MatrixXd &cov,
                                           int num_samples)
{
    const int dim = mean.size();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
    Eigen::MatrixXd transform = eigenSolver.eigenvectors() *
                                eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();

    // Set up a standard normal generator.
    std::mt19937 rng(42);
    std::normal_distribution<double> standard_normal(0.0, 1.0);

    // Create matrix to hold samples: each row is a sample.
    Eigen::MatrixXd samples(num_samples, dim);
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd z(dim);
        for (int d = 0; d < dim; ++d) {
            z(d) = standard_normal(rng);
        }
        samples.row(i) = mean + transform * z;
    }
    return samples;
}

static void BM_RLE_MI(benchmark::State& state) {
    // Here, N is both the number of dimensions and the number of correlated normals.
    const int N = state.range(0);
    const int num_samples = 10000; // number of samples drawn from the multivariate normal

    // Setup random generators.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> mean_dist(0.0, 100.0);
    std::uniform_real_distribution<double> variance_dist(10.0, 100.0);

    // Generate a random mean vector (size N).
    Eigen::VectorXd mean(N);
    for (int i = 0; i < N; ++i) {
        mean(i) = mean_dist(rng);
    }

    // Generate a random variance vector (size N).
    Eigen::VectorXd variance(N);
    for (int i = 0; i < N; ++i) {
        variance(i) = variance_dist(rng);
    }

    // Generate a random correlation matrix (N x N).
    // One common method is to generate an N x N matrix A with normally distributed entries,
    // then form the Gram matrix A * A^T, which is positive definite.
    Eigen::MatrixXd A(N, N);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A(i, j) = normal_dist(rng);
    Eigen::MatrixXd corr = A * A.transpose();

    // Normalize the Gram matrix to obtain a proper correlation matrix.
    for (int i = 0; i < N; ++i) {
        double diag = std::sqrt(corr(i, i));
        for (int j = 0; j < N; ++j) {
            double diag_j = std::sqrt(corr(j, j));
            corr(i, j) /= (diag * diag_j);
        }
        corr(i, i) = 1.0;  // Ensure the diagonal is exactly 1.
    }

    // Convert the correlation matrix and variances into a covariance matrix.
    Eigen::MatrixXd cov = correlationToCovariance(corr, variance);

    // Sample from the N-dimensional multivariate normal.
    // The returned samples matrix has shape: num_samples x N.
    Eigen::MatrixXd samples = sampleMultivariateNormal2(mean, cov, num_samples);

    // For each of the N dimensions, quantise (round) the values and
    // pack them into a run-length encoded vector.
    std::vector<std::vector<std::pair<int, int>>> all_rle;
    all_rle.reserve(N);
    for (int col = 0; col < N; ++col) {
        std::vector<int> quantised;
        quantised.reserve(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            // Quantise by rounding.
            quantised.push_back(static_cast<int>(std::round(samples(i, col))));
        }
        auto rle = runLengthEncoding(quantised);
        all_rle.push_back(std::move(rle));
    }

    std::cout << all_rle.size() << std::endl;

    // In the benchmark loop, simply aggregate some property of the pre-generated RLE
    // to simulate processing and to prevent the compiler from optimizing the code away.
    for (auto _ : state) {
        int total_runs = 0;
        for (const auto &rle : all_rle) {
            total_runs += static_cast<int>(rle.size());
        }
        benchmark::DoNotOptimize(total_runs);
    }
    state.SetComplexityN(N);
}

// BENCHMARK(BM_MI)->Range(16, 1 << 16)->Complexity();
BENCHMARK(BM_RLE_MI)->Range(16, 1 << 16)->Complexity();

BENCHMARK_MAIN();
