#include <sys/qos.h>

#include <cmath>
#include <numeric>
#include <vector>

#ifdef ENABLE_BENCHMARK
#include <benchmark/benchmark.h>
#endif

#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "kdtree.hpp"
#include "mutual_information.hpp"
#include "mvn.hpp"
#include "tests.hpp"
#include "utils.hpp"

Eigen::MatrixXd sampleIndependentNormals(const Eigen::VectorXd& mean,
                                         const Eigen::VectorXd& variance,
                                         int num_samples) {
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

#ifdef ENABLE_BENCHMARK

static void BM_MI(benchmark::State& state) {
    Eigen::VectorXd mean(2);
    mean << 40.0, 150.0;

    Eigen::VectorXd variance(2);
    variance << 30.0, 50.0;

    Eigen::MatrixXd corr(2, 2);
    corr << 1.0, -0.7, -0.7, 1.0;

    Eigen::MatrixXd cov = correlationToCovariance(corr, variance);

    int num_samples = 10000;

    Eigen::MatrixXd samples = sampleMultivariateNormal(mean, cov, num_samples);

    Eigen::VectorXd std_dev = variance.array().sqrt();

    for (auto _ : state) {
        std::vector<float> results(state.range(0));
        for (size_t i = 0; i < results.size(); ++i) {
            double mi = mutual_information_quantised(
                mean(0), std_dev(0), mean(1), std_dev(1), samples, 10);

            results[i] = mi;
        }
        benchmark::DoNotOptimize(results);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_RLE_MI(benchmark::State& state) {
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        unsigned int numCores = std::thread::hardware_concurrency();
        if (numCores == 0) {
            numCores =
                1;  // Fallback if hardware_concurrency cannot detect cores.
        }
        omp_set_num_threads(static_cast<int>(numCores / 2));
    }

    // Here, N is both the number of dimensions and the number of correlated
    // normals.
    const int N = state.range(0);
    const int num_samples =
        10000;  // number of samples drawn from the multivariate normal

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

    Eigen::MatrixXi samples =
        sampleIndependentNormals(mean, variance, num_samples).cast<int>();

    for (auto _ : state) {
        Eigen::MatrixXd mi = mutual_information_rle(samples, mean, variance);

        benchmark::DoNotOptimize(mi);
    }
    state.SetComplexityN(N);
}

// BENCHMARK(BM_MI)->Range(16, 1 << 16)->Complexity();
BENCHMARK(BM_RLE_MI)->Range(16, 1 << 14)->Complexity();

BENCHMARK_MAIN();

#else

int main() {
    // Here, N is both the number of dimensions and the number of correlated
    // normals.
    const int N = 2;
    const int num_samples =
        100000;  // number of samples drawn from the multivariate normal

    // Setup random generators.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> mean_dist(99., 101.);
    std::uniform_real_distribution<double> variance_dist(99., 100.0);

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

    Eigen::MatrixXd samples =
        sampleIndependentNormals(mean, variance, num_samples);

    std::cout << samples.rows() << " " << samples.cols() << std::endl;

    // Standard 2d

    // double mi_2d = mutual_information_normal(mean(0), std::sqrt(variance(0)),
    // mean(1), std::sqrt(variance(1)), samples);

    Eigen::VectorXd f1 = samples.col(0);
    Eigen::VectorXd f2 = samples.col(1);

    Eigen::VectorXd std_dev = variance.array().sqrt();

    double mi_2d = mutual_information_normal(mean(0), std_dev(0), mean(1),
                                             std_dev(1), f1, f2, 25);

    std::cout << "MI " << mi_2d << std::endl;

    double mi_2dq = mutual_information_quantised(mean(0), std_dev(0),
    mean(1), std_dev(1), f1, f2, 25);

    std::cout << "MI Q " << mi_2dq << std::endl;

    std::cout << samples.rows() << " " << samples.cols() << std::endl;

    // Should be equivalent
    std::vector<Point<int>> point_samples = convertSamplesToPointsQuantised(samples);
    // std::vector<Point<int>> point_samples = convertSamplesToPointsQuantised(f1, f2);

    double mi_2dq_t = mutual_information_normal(mean(0), std_dev(0), mean(1),
                                             std_dev(1), point_samples, 25);

    std::cout << "MI Q test " << mi_2dq_t << std::endl;

    // Standard multi-dimensional
    Eigen::MatrixXd mi_nd = mutual_information_normal(samples, mean, std_dev, 25);

    std::cout << "MI ND" << mi_nd << std::endl;

    // The direct cast means that these probably aren't going to be the same
    Eigen::MatrixXi samples_int = samples.cast<int>();

    // Quantised multi-dimensional
    Eigen::MatrixXd mi_ndq =
        mutual_information_normal(samples_int, mean, std_dev, 25);

    std::cout << "MI ND Q" << mi_ndq << std::endl;

    // RLE multi-dimensional
}

#endif
