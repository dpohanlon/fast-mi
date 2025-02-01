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

    int num_samples = 100000;

    Eigen::MatrixXd samples = sampleMultivariateNormal(mean, cov, num_samples);

    Eigen::VectorXd std_dev = variance.array().sqrt();

    for (auto _ : state) {
        std::vector<float> results(state.range(0));
        for (size_t i = 0; i < results.size(); ++i) {

            double mi = mutual_information_quantised(mean(0), std_dev(0), mean(1), std_dev(1), samples, 100);

            results[i] = mi;
        }
        benchmark::DoNotOptimize(results);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_MI)->Range(16, 1 << 16)->Complexity();

BENCHMARK_MAIN();
