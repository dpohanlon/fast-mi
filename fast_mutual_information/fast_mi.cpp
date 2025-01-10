#include <vector>
#include <cmath>
#include <numeric>
#include <benchmark/benchmark.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <functional>

#include "kdtree.hpp"
#include "tests.hpp"
#include "mvn.hpp"

/**
 * @brief Converts an Eigen::MatrixXd of multivariate normal samples to a std::vector<Point>.
 *
 * Each column of the 'samples' matrix represents one sample. This function assumes
 * that each sample has exactly two dimensions corresponding to 'x' and 'y'.
 *
 * @param samples        The Eigen::MatrixXd containing the samples (2 x num_samples).
 * @param rounding_mode  The mode of rounding: "round", "floor", "ceil", or "truncate".
 * @return std::vector<Point> A vector of Points with integer coordinates.
 *
 * @throws std::invalid_argument If the samples matrix does not have 2 rows.
 */
std::vector<Point> convertSamplesToPoints(
    const Eigen::MatrixXd& samples,
    const std::string& rounding_mode = "round")
{
    // Ensure that the samples matrix has exactly 2 rows for x and y
    if (samples.rows() != 2) {
        throw std::invalid_argument("Samples matrix must have exactly 2 rows for x and y coordinates.");
    }

    int num_samples = static_cast<int>(samples.cols());
    std::vector<Point> points;
    points.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        double x_double = samples(0, i);
        double y_double = samples(1, i);
        int x_int, y_int;

        // Convert double to int based on the rounding mode
        if (rounding_mode == "round") {
            x_int = static_cast<int>(std::round(x_double));
            y_int = static_cast<int>(std::round(y_double));
        }
        else if (rounding_mode == "floor") {
            x_int = static_cast<int>(std::floor(x_double));
            y_int = static_cast<int>(std::floor(y_double));
        }
        else if (rounding_mode == "ceil") {
            x_int = static_cast<int>(std::ceil(x_double));
            y_int = static_cast<int>(std::ceil(y_double));
        }
        else if (rounding_mode == "truncate") {
            x_int = static_cast<int>(x_double); // Truncates towards zero
            y_int = static_cast<int>(y_double);
        }
        else {
            throw std::invalid_argument("Invalid rounding_mode. Choose from 'round', 'floor', 'ceil', or 'truncate'.");
        }

        // Create a Point and add to the vector
        points.emplace_back(Point{ x_int, y_int });
    }

    return points;
}

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

int main() {

    Eigen::VectorXd mean(2);
    mean << 1.0, 2.0;

    Eigen::VectorXd variances(2);
    variances << 50.0, 10.0;

    Eigen::MatrixXd corr(2, 2);
    corr <<  1.0,  0.5,
             0.5,  1.0;

    Eigen::MatrixXd cov = correlationToCovariance(corr, variances);

    int num_samples = 100000;

    Eigen::MatrixXd samples = sampleMultivariateNormal(mean, cov, num_samples);

    std::vector<Point> point_samples = convertSamplesToPoints(samples);

    // Lambda functions to wrap the PMFs
    auto p_x_func = [&](int x) -> double {
        return normal_pdf(x, mean(0), std::sqrt(variances(0)));
    };
    auto p_y_func = [&](int y) -> double {
        return normal_pdf(y, mean(1), std::sqrt(variances(1)));
    };

    KDTree tree(point_samples, 10);

    // Compute mutual information
    double mi = tree.compute_mutual_information(p_x_func, p_y_func);

    double analyticalMI = -0.5 * std::log(1 - std::pow(corr(0, 1), 2));

    std::cout << analyticalMI <<  " " << mi << std::endl;

    return 0;
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
