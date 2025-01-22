#include <vector>
#include <cmath>
#include <numeric>

// #ifdef ENABLE_BENCHMARK
// #include <benchmark/benchmark.h>
// #endif

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <functional>

#include "kdtree.hpp"
#include "tests.hpp"
#include "mvn.hpp"
#include "utils.hpp"
#include "mutual_information.hpp"

// int main() {
//     // Example data generation (replace with your actual data)
//     std::vector<Point> data = generate_correlated_data();

//     // Initialize kd-tree with max 2 points per leaf
//     KDTree tree(data, 10);

//     // Define Negative Binomial PMFs for X and Y
//     NegativeBinomial nb_x(5.0, 0.5);
//     NegativeBinomial nb_y(5.0, 0.5);

//     // Lambda functions to wrap the PMFs
//     auto p_x_func = [&](int x) -> double {
//         return nb_x.pmf(x);
//     };
//     auto p_y_func = [&](int y) -> double {
//         return nb_y.pmf(y);
//     };

//     // Compute mutual information
//     double mi = tree.compute_mutual_information(p_x_func, p_y_func);

//     std::cout << "Mutual Information: " << mi << " bits\n";
//     return 0;
// }

// int main() {

//     Eigen::VectorXd mean(2);
//     mean << 10.0, 50.0;

//     Eigen::VectorXd variance(2);
//     variance << 5.0, 15.0;

//     Eigen::MatrixXd corr(2, 2);
//     corr <<  1.0,  -0.7,
//             -0.7,  1.0;

//     Eigen::MatrixXd cov = correlationToCovariance(corr, variance);

//     int num_samples = 100000;

//     Eigen::MatrixXd samples = sampleMultivariateNormal(mean, cov, num_samples);

//     double mi = mutual_information_normal(mean(0), std::sqrt(variance(0)), mean(1), std::sqrt(variance(1)), samples);

//     double analyticalMI = -0.5 * std::log(1 - std::pow(corr(0, 1), 2));

//     std::cout << analyticalMI <<  " " << mi << std::endl;

//     return 0;
// }

int main() {

    Eigen::VectorXd mean(2);
    mean << 10.0, 50.0;

    Eigen::VectorXd variance(2);
    variance << 5.0, 15.0;

    Eigen::MatrixXd corr(2, 2);
    corr <<  1.0,  -0.7,
            -0.7,  1.0;

    const double step = 0.01;

    std::vector<double> rho_vec;
    std::vector<double> mi_analytical_vec;
    std::vector<double> mi_tree_vec;

    for (double rho = -0.99; rho <= 1.0; rho += step) {

        rho_vec.push_back(rho);

        corr(0, 1) = rho;
        corr(1, 0) = rho;

        Eigen::MatrixXd cov = correlationToCovariance(corr, variance);

        int num_samples = 1000;

        Eigen::MatrixXd samples = sampleMultivariateNormal(mean, cov, num_samples);

        double mi = mutual_information_normal(mean(0), std::sqrt(variance(0)), mean(1), std::sqrt(variance(1)), samples);

        double analyticalMI = -0.5 * std::log(1 - std::pow(corr(0, 1), 2));

        mi_analytical_vec.push_back(analyticalMI);
        mi_tree_vec.push_back(mi);

    }

    std::string csv_filename = "mi_1k.csv";

    std::ofstream file(csv_filename);

    file << "rho,mi,mi_tree\n";

    file << std::fixed << std::setprecision(6);

    for (int i = 0; i < rho_vec.size(); i++) {
        file << rho_vec[i] << "," << mi_analytical_vec[i] << "," << mi_tree_vec[i] << "\n";
    }

    file.close();

}

// static void BM_MI(benchmark::State& state) {

//     std::vector<Point> data = generate_correlated_data(state.range(0));

//     for (auto _ : state) {
//         std::vector<float> results(state.range(0));
//         for (size_t i = 0; i < results.size(); ++i) {

//             KDTree tree(data, 10);

//             NegativeBinomial nb_x(5.0, 0.5);
//             NegativeBinomial nb_y(5.0, 0.5);

//             auto p_x_func = [&](int x) -> double {
//                 return nb_x.pmf(x);
//             };
//             auto p_y_func = [&](int y) -> double {
//                 return nb_y.pmf(y);
//             };

//             results[i] = tree.compute_mutual_information(p_x_func, p_y_func);
//         }
//         benchmark::DoNotOptimize(results);
//     }
//     state.SetComplexityN(state.range(0));
// }

// BENCHMARK(BM_MI)->Range(16, 1 << 16)->Complexity();

// BENCHMARK_MAIN();
