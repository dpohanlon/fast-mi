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

Eigen::MatrixXd sampleIndependentNormals(const Eigen::VectorXd &mean,
                                           const Eigen::VectorXd &variance,
                                           int num_samples)
{
    const int dim = mean.size();
    Eigen::MatrixXd samples(num_samples, dim);
    std::mt19937 rng(42);
    for (int d = 0; d < dim; ++d) {
        // Create a normal distribution for the d-th feature.
        double m = mean(d);
        double sd = std::sqrt(variance(d));
        std::normal_distribution<double> norm(m, sd);
        for (int i = 0; i < num_samples; ++i) {
            samples(i, d) = norm(rng);
        }
    }
    return samples;
}

static void BM_RLE_MI(benchmark::State& state) {
    // Here, N is both the number of dimensions and the number of correlated normals.
    const int N = state.range(0);
    const int num_samples = 10000; // number of samples drawn from the multivariate normal

    // Setup random generators.
    std::mt19937 rng(42);
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

    Eigen::MatrixXd samples = sampleIndependentNormals(mean, variance, num_samples);

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

    // In the benchmark loop, simply aggregate some property of the pre-generated RLE
    // to simulate processing and to prevent the compiler from optimizing the code away.
    for (auto _ : state) {

        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                double mi = mutual_information_quantised_rle(mean(i), std::sqrt(variance(i)), mean(j), std::sqrt(variance(j)), all_rle[i], all_rle[j], num_samples);
                benchmark::DoNotOptimize(mi);
            }
        }

    }
    state.SetComplexityN(N);
}

BENCHMARK(BM_MI)->Range(16, 1 << 16)->Complexity();
BENCHMARK(BM_RLE_MI)->Range(16, 1 << 16)->Complexity();

BENCHMARK_MAIN();
