#pragma once

#include <numeric>
#include <vector>
#include <limits>
#include <algorithm>

#include <Eigen/Dense>

#include "point.hpp"
#include "mvn.hpp"

std::vector<std::pair<int, int>> runLengthEncoding(std::vector<int> & points)
{

    if (points.empty()) return {};

    auto sorted_points = points;

    // Here maybe a radix sort?
    std::sort(sorted_points.begin(), sorted_points.end());

    std::vector<std::pair<int, int>> result;
    result.reserve(points.size() / 2);

    int current = sorted_points[0];
    int count = 1;

    for (std::size_t i = 1; i < sorted_points.size(); ++i) {
        if (sorted_points[i] == current) {
            ++count;
        } else {
            result.emplace_back(current, count);
            current = sorted_points[i];
            count = 1;
        }
    }

    result.emplace_back(current, count);

    return result;

}

std::vector<std::pair<int, int>> runLengthEncoding(const Eigen::Ref<const Eigen::VectorXi>& points)
{
    std::vector<int> eigen_vector(points.data(), points.data() + points.size());

    return runLengthEncoding(eigen_vector);
}

// This function takes two run-length encoded vectors of ints, where each
// element is a pair {value, count}. It "zips" the two sequences into a sequence
// of Point<int> (with x from the first sequence and y from the second) and produces
// a run-length encoded vector of these points.
std::vector<std::pair<Point<int>, int>>
runLengthDecoding(std::vector<std::pair<int, int>> & first,
                    std::vector<std::pair<int, int>> & second)
{
    std::vector<std::pair<Point<int>, int>> result;

    // Indexes into the first and second encoded vectors.
    std::size_t i = 0, j = 0;

    // Remaining count for the current run in each vector.
    int remain1 = (i < first.size() ? first[i].second : 0);
    int remain2 = (j < second.size() ? second[j].second : 0);

    // Process until we have exhausted one of the sequences.
    while (i < first.size() && j < second.size()) {
        // Form the current point from the current values of the two sequences.
        Point<int> pt { first[i].first, second[j].first };

        // The next run in the output can only be as long as the smaller number
        // of available occurrences in the two input runs.
        int run = std::min(remain1, remain2);

        // If the last point in the result is the same as the current one,
        // then extend that run. Otherwise, add a new run.
        if (!result.empty() &&
            result.back().first.x == pt.x &&
            result.back().first.y == pt.y) {
            result.back().second += run;
        } else {
            result.push_back({pt, run});
        }

        // Decrease the counts from each run.
        remain1 -= run;
        remain2 -= run;

        // Move to the next run in the first sequence if the current run is exhausted.
        if (remain1 == 0) {
            ++i;
            if (i < first.size()) {
                remain1 = first[i].second;
            }
        }
        // Move to the next run in the second sequence if the current run is exhausted.
        if (remain2 == 0) {
            ++j;
            if (j < second.size()) {
                remain2 = second[j].second;
            }
        }
    }

    return result;
}

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

// Could template specialise, but it's probably not worth it

std::vector<Point<int>> convertSamplesToPointsQuantised(
    const Eigen::MatrixXd& samples,
    const std::string& rounding_mode = "round")
{
    // Ensure that the samples matrix has exactly 2 rows for x and y
    if (samples.rows() != 2) {
        throw std::invalid_argument("Samples matrix must have exactly 2 rows for x and y coordinates.");
    }

    int num_samples = static_cast<int>(samples.cols());
    std::vector<Point<int>> points;
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

        points.emplace_back(Point<int>{ x_int, y_int });
    }

    return points;
}

// Change the function signature to accept two Eigen::VectorXd objects for x and y.
std::vector<Point<int>> convertSamplesToPointsQuantised(
    const Eigen::VectorXd& samples1,
    const Eigen::VectorXd& samples2,
    const std::string& rounding_mode = "round")
{

    if (samples1.size() != samples2.size()) {
        throw std::invalid_argument("Samples vectors must have the same length.");
    }

    int num_samples = static_cast<int>(samples1.size());
    std::vector<Point<int>> points;
    points.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        double x_double = samples1(i);
        double y_double = samples2(i);
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

        points.emplace_back(Point<int>{ x_int, y_int });
    }

    return points;
}


// Make this templated, but have to use the 'long' form of the Eigen type names
std::vector<Point<double>> convertSamplesToPoints(const Eigen::MatrixXd& samples)
{
    // Ensure that the samples matrix has exactly 2 rows for x and y
    if (samples.cols() != 2) {
        throw std::invalid_argument("Samples matrix must have exactly 2 rows for x and y coordinates.");
    }

    int num_samples = static_cast<int>(samples.rows());
    std::vector<Point<double>> points(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        double x = samples(i, 0);
        double y = samples(i, 1);

        points[i] = Point<double>{ x, y };
    }

    return points;
}

// Function to transform samples from normal to uniform distribution
Eigen::MatrixXd transformToUniform(const Eigen::MatrixXd& samples, const Eigen::VectorXd& mean, const Eigen::VectorXd& std_dev) {
    Eigen::MatrixXd uniform_samples(samples.rows(), samples.cols());

    // std::cout << samples.rows() <<  " " << samples.cols() << std::endl;
    // 10, 2

    // Cells
    for (int i = 0; i < samples.rows(); ++i) {
        // Genes
        for (int j = 0; j < samples.cols(); ++j) {
            uniform_samples(i, j) = normal_cdf(samples(i, j), mean(j), std_dev(j));
        }
    }

    return uniform_samples;
}

void clamp_uniform_samples(std::vector<Point<double>>& points) {
    constexpr double CLAMP_EPS = 1e-12;
    for (auto &pt : points) {
        pt.x = std::clamp(pt.x, CLAMP_EPS, 1.0 - CLAMP_EPS);
        pt.y = std::clamp(pt.y, CLAMP_EPS, 1.0 - CLAMP_EPS);
    }
}

void clamp_uniform_samples(Eigen::MatrixXd& data) {
    data = data.cwiseMax(0.0).cwiseMin(1.0);
}

template <typename T>
struct Bounds {

    Bounds() = default;

    Bounds(T min_x, T max_x, T min_y, T max_y);

    T min_x = std::numeric_limits<T>::max();
    T max_x = std::numeric_limits<T>::min();
    T min_y = std::numeric_limits<T>::max();
    T max_y = std::numeric_limits<T>::min();
};

template <typename T>
Bounds<T>::Bounds(T min_x, T max_x, T min_y, T max_y) : min_x(min_x), max_x(max_x), min_y(min_y), max_y(max_y) {}

template <typename T>
Bounds<T> get_bounds(const std::vector<Point<T>>& points) {
  if (points.empty()) {
    // Handle empty case, e.g., return default values or throw an exception.
    return Bounds<T>();
  }

  T min_x = std::numeric_limits<T>::max();
  T max_x = std::numeric_limits<T>::min();
  T min_y = std::numeric_limits<T>::max();
  T max_y = std::numeric_limits<T>::min();

  Bounds<T> bounds;

  for (const auto& point : points) {
    bounds.min_x = std::min(min_x, point.x);
    bounds.max_x = std::max(max_x, point.x);
    bounds.min_y = std::min(min_y, point.y);
    bounds.max_y = std::max(max_y, point.y);
  }

  return bounds;
}

Bounds<int> get_bounds(const std::vector<std::pair<Point<int>, int>>& points) {
  if (points.empty()) {
    // Handle empty case as needed; here we return a default-constructed Bounds.
    return Bounds<int>();
  }

  int min_x = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int min_y = std::numeric_limits<int>::max();
  int max_y = std::numeric_limits<int>::min();

  for (const auto& pair : points) {
    const Point<int>& pt = pair.first;
    min_x = std::min(min_x, pt.x);
    max_x = std::max(max_x, pt.x);
    min_y = std::min(min_y, pt.y);
    max_y = std::max(max_y, pt.y);
  }

  return Bounds<int>(min_x, max_x, min_y, max_y);
}
