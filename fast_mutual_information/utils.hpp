#pragma once

#include <numeric>

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

// Specialise on return type

// std::vector<IPoint> convertSamplesToPoints(
//     const Eigen::MatrixXd& samples,
//     const bool quantise = false,
//     const std::string& rounding_mode = "round")
// {
//     // Ensure that the samples matrix has exactly 2 rows for x and y
//     if (samples.rows() != 2) {
//         throw std::invalid_argument("Samples matrix must have exactly 2 rows for x and y coordinates.");
//     }

//     int num_samples = static_cast<int>(samples.cols());
//     std::vector<IPoint> points;
//     points.reserve(num_samples);

//     for (int i = 0; i < num_samples; ++i) {
//         double x_double = samples(0, i);
//         double y_double = samples(1, i);
//         int x_int, y_int;

//         // Convert double to int based on the rounding mode
//         if (rounding_mode == "round") {
//             x_int = static_cast<int>(std::round(x_double));
//             y_int = static_cast<int>(std::round(y_double));
//         }
//         else if (rounding_mode == "floor") {
//             x_int = static_cast<int>(std::floor(x_double));
//             y_int = static_cast<int>(std::floor(y_double));
//         }
//         else if (rounding_mode == "ceil") {
//             x_int = static_cast<int>(std::ceil(x_double));
//             y_int = static_cast<int>(std::ceil(y_double));
//         }
//         else if (rounding_mode == "truncate") {
//             x_int = static_cast<int>(x_double); // Truncates towards zero
//             y_int = static_cast<int>(y_double);
//         }
//         else {
//             throw std::invalid_argument("Invalid rounding_mode. Choose from 'round', 'floor', 'ceil', or 'truncate'.");
//         }

//         // Create a Point and add to the vector
//         points.emplace_back(IPoint{ x_int, y_int });

//         // Create a Point and add to the vector
//         points.emplace_back(IPoint{ x_double, y_double });
//     }

//     return points;
// }

std::vector<RPoint> convertSamplesToPoints(const Eigen::MatrixXd& samples)
{
    // Ensure that the samples matrix has exactly 2 rows for x and y
    if (samples.rows() != 2) {
        throw std::invalid_argument("Samples matrix must have exactly 2 rows for x and y coordinates.");
    }

    int num_samples = static_cast<int>(samples.cols());
    std::vector<RPoint> points;
    points.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        double x = samples(0, i);
        double y = samples(1, i);

        points.emplace_back(RPoint{ x, y });
    }

    return points;
}

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
