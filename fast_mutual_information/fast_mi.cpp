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

// Function to transform samples from normal to uniform distribution
Eigen::MatrixXd transformToUniform(const Eigen::MatrixXd& samples, const Eigen::VectorXd& mean, const Eigen::VectorXd& std_dev) {
    Eigen::MatrixXd uniform_samples(samples.rows(), samples.cols());

    for (int i = 0; i < samples.cols(); ++i) {
        for (int j = 0; j < samples.rows(); ++j) {
            uniform_samples(j, i) = normal_cdf(samples(j, i), mean(j), std_dev(j));
        }
    }

    return uniform_samples;
}

void clamp_uniform_samples(std::vector<Point>& points) {
    constexpr double CLAMP_EPS = 1e-12;
    for (auto &pt : points) {
        pt.x = std::clamp(pt.x, CLAMP_EPS, 1.0 - CLAMP_EPS);
        pt.y = std::clamp(pt.y, CLAMP_EPS, 1.0 - CLAMP_EPS);
    }
}

int main() {

    Eigen::VectorXd mean(2);
    mean << 30.0, 150.0;
    // mean << 0.0, 0.0;

    Eigen::VectorXd variance(2);
    variance << 5.0, 15.0;
    // variance << 1.0, 1.0;

    Eigen::MatrixXd corr(2, 2);
    corr <<  1.0,  -0.7,
            -0.7,  1.0;

    const double step = 0.01;

    std::vector<double> rho_vec;
    std::vector<double> mi_analytical_vec;
    std::vector<double> mi_tree_vec;

    for (double rho = -0.99; rho <= 1.0; rho += step) {

        // double rho = 0.0;

        rho_vec.push_back(rho);

        corr(0, 1) = rho;
        corr(1, 0) = rho;

        Eigen::MatrixXd cov = correlationToCovariance(corr, variance);

        int num_samples = 10000;

        Eigen::MatrixXd samples = sampleMultivariateNormal(mean, cov, num_samples);

        // Transform the samples to a uniform distribution
        Eigen::VectorXd std_dev = variance.array().sqrt();
        Eigen::MatrixXd uniform_samples = transformToUniform(samples, mean, std_dev);

        // Convert to vector of Points for the mutual information function
        std::vector<Point> point_samples = convertSamplesToPoints(uniform_samples);

        clamp_uniform_samples(point_samples);

        double mi = mutual_information_normal(point_samples);

        double analyticalMI = -0.5 * std::log(1. - std::pow(corr(0, 1), 2.));

        mi_analytical_vec.push_back(analyticalMI);
        mi_tree_vec.push_back(mi);

    }

    std::string csv_filename = "mi_100k.csv";

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
