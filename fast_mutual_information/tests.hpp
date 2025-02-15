#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "kdtree.hpp"

// Function to generate a Poisson-distributed random variable using Knuth's
// algorithm
int generate_poisson(double lambda, std::mt19937& rng) {
    if (lambda < 0) return 0;
    if (lambda == 0) return 0;

    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    double L = std::exp(-lambda);
    int k = 0;
    double p = 1.0;

    while (p > L) {
        k++;
        double u = uniform_dist(rng);
        p *= u;
    }

    return k - 1;
}

std::vector<Point<double>> generate_correlated_data(int num_points = 1000) {
    // Seed the random number generator with a high-resolution clock
    std::mt19937 rng(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // Parameters for Negative Binomial distributions for X and Y
    // These parameters define the marginals p(x) and p(y)
    // r: number of failures until the experiment is stopped
    // p: probability of success in each trial

    // For X
    double r_x = 5.0;  // Number of failures
    double p_x = 0.5;  // Probability of success

    // For Y
    double r_y = 7.0;  // Number of failures
    double p_y = 0.6;  // Probability of success

    // Parameters for the shared Gamma distribution to induce correlation
    // You can adjust the shared component to control the correlation strength
    double shared_alpha = 2.0;                 // Shape parameter
    double shared_beta_x = p_x / (1.0 - p_x);  // Rate parameter for X
    double shared_beta_y = p_y / (1.0 - p_y);  // Rate parameter for Y

    // Gamma distribution for shared component Z
    std::gamma_distribution<double> gamma_shared_x(r_x, shared_beta_x);
    std::gamma_distribution<double> gamma_shared_y(r_y, shared_beta_y);

    // Vector to store the generated points
    std::vector<Point<double>> dataset;
    dataset.reserve(num_points);

    for (int i = 0; i < num_points; ++i) {
        // Generate shared Gamma variable for X and Y
        double z_shared_x = gamma_shared_x(rng);
        double z_shared_y = gamma_shared_y(rng);

        // Generate X and Y from Poisson distributions with rate parameters
        // derived from Gamma variables
        double x = generate_poisson(z_shared_x, rng);
        double y = generate_poisson(z_shared_y, rng);

        // Store the generated point
        dataset.push_back(Point<double>{x, y});
    }

    return dataset;
}
