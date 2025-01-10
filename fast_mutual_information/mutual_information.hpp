#pragma once

#include <Eigen/Dense>

#include "utils.hpp"
#include "mvn.hpp"
#include "kdtree.hpp"

// Set up with a class, configure, then run MI calculation

// TODO: Take Eigen vectors of means and variances
double mutual_information_normal(double mean1, double std_dev1, double mean2, double std_dev2, std::vector<Point> & data, int min_pop = 10)
{
    auto p_x_func = [&](int x) -> double {
        return normal_pdf(x, mean1, std_dev1);
    };
    auto p_y_func = [&](int y) -> double {
        return normal_pdf(y, mean2, std_dev2);
    };

    KDTree tree(data, min_pop);

    return tree.compute_mutual_information(p_x_func, p_y_func);

}

// Mutual information with normally distributed marginals
double mutual_information_normal(double mean1, double std_dev1, double mean2, double std_dev2, Eigen::MatrixXd & data, int min_pop = 10)
{
    // Real value input - quantise first

    std::vector<Point> point_samples = convertSamplesToPoints(data);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2, point_samples, min_pop);
}
