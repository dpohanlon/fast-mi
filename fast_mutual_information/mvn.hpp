#pragma once

#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

/**
 * @brief Converts a correlation matrix and a vector of variances to a
 * covariance matrix.
 *
 * @param correlationMatrix The correlation matrix (n x n), must be symmetric
 * with 1s on the diagonal.
 * @param variances        The vector of variances (size n).
 * @return Eigen::MatrixXd   The resulting covariance matrix (n x n).
 *
 * @throws std::invalid_argument If the correlation matrix is not square or if
 * its size does not match the variances vector.
 */
Eigen::MatrixXd correlationToCovariance(
    const Eigen::MatrixXd& correlationMatrix,
    const Eigen::VectorXd& variances) {
    // Ensure the correlation matrix is square
    if (correlationMatrix.rows() != correlationMatrix.cols()) {
        throw std::invalid_argument("Correlation matrix must be square.");
    }

    // Ensure the size of variances matches the correlation matrix
    if (correlationMatrix.rows() != variances.size()) {
        throw std::invalid_argument(
            "Size of variances vector must match the correlation matrix "
            "dimensions.");
    }

    // Compute standard deviations by taking the square root of variances
    Eigen::VectorXd stddev = variances.array().sqrt();

    // Create a diagonal matrix of standard deviations
    Eigen::MatrixXd D = stddev.asDiagonal();

    // Compute the covariance matrix: Cov = D * Corr * D
    Eigen::MatrixXd covarianceMatrix = D * correlationMatrix * D;

    return covarianceMatrix;
}

/**
 * @brief Computes the Probability Density Function (PDF) of a univariate normal
 * distribution.
 *
 * @param x     The point at which to evaluate the PDF.
 * @param mean  The mean (\mu) of the distribution.
 * @param stddev The standard deviation (\sigma) of the distribution.
 * @return double The value of the PDF at point x.
 *
 * @throws std::invalid_argument If the standard deviation is non-positive.
 */
double normal_pdf(double x, double mean, double stddev) {
    if (stddev <= 0.0) {
        throw std::invalid_argument("Standard deviation must be positive.");
    }

    boost::math::normal dist(mean, stddev);
    return boost::math::pdf(dist, x);
}

// Cumulative Distribution Function (CDF) of the normal distribution
template <typename T>
T normal_cdf(T x, double mean, double stddev) {

    if (stddev <= 0.0) {
        throw std::invalid_argument("Standard deviation must be positive.");
    }

    boost::math::normal dist(mean, stddev);
    return boost::math::cdf(dist, x);
}

// Inverse Cumulative Distribution Function (Inverse CDF) or Quantile Function
// of the normal distribution
double normal_icdf(double p, double mean, double stddev) {

    if (stddev <= 0.0) {
        throw std::invalid_argument("Standard deviation must be positive.");
    }
    if (p < 0.0 || p >= 1.0) {
        throw std::invalid_argument("Probability p must be between 0 and 1.");
    }

    boost::math::normal dist(mean, stddev);
    return boost::math::quantile(dist, p);
}

/**
 * \brief Samples points from a multivariate normal distribution using Eigen.
 *
 * \param mean        The mean vector (size n).
 * \param cov         The covariance matrix (n x n).
 * \param num_samples The number of samples to draw.
 * \return A matrix of size (n x num_samples). Each column is one sample.
 */
Eigen::MatrixXd sampleMultivariateNormal(const Eigen::VectorXd& mean,
                                         const Eigen::MatrixXd& cov,
                                         int num_samples) {

    const int n = static_cast<int>(mean.size());

    Eigen::LLT<Eigen::MatrixXd> lltOfCov(cov);
    if (lltOfCov.info() == Eigen::NumericalIssue) {
        throw std::runtime_error(
            "Cholesky decomposition failed (covariance not positive "
            "definite?)");
    }
    Eigen::MatrixXd L = lltOfCov.matrixL();  // L is lower-triangular

    std::mt19937 rng(static_cast<unsigned long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXd result(n, num_samples);

    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd z(n);
        for (int j = 0; j < n; ++j) {
            z(j) = dist(rng);
        }

        Eigen::VectorXd x_prime = L * z;

        Eigen::VectorXd x = mean + x_prime;

        result.col(i) = x;
    }

    return result;
}
