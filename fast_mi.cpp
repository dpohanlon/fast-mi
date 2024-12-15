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

static void BM_MI(benchmark::State& state) {

    std::vector<int> k_vals(state.range(0));
    std::iota(k_vals.begin(), k_vals.end(), 0);

    std::vector<Point> data = generate_correlated_data(10000);

    for (auto _ : state) {
        std::vector<float> results(k_vals.size());
        for (size_t i = 0; i < k_vals.size(); ++i) {

            KDTree tree(data, 10);

            NegativeBinomial nb_x(5.0, 0.5);
            NegativeBinomial nb_y(5.0, 0.5);

            auto p_x_func = [&](int x) -> double {
                return nb_x.pmf(x);
            };
            auto p_y_func = [&](int y) -> double {
                return nb_y.pmf(y);
            };

            results[i] = tree.compute_mutual_information(p_x_func, p_y_func);
        }
        benchmark::DoNotOptimize(results);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_MI)->Range(16, 1 << 16)->Complexity();

BENCHMARK_MAIN();
