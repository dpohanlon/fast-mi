#pragma once

#include<functional>

#include <Eigen/Dense>
#include "fast_negative_binomial/fast_nb.hpp"

#include "utils.hpp"
#include "mvn.hpp"
#include "kdtree.hpp"
#include "copula.hpp"
// Set up with a class, configure, then run MI calculation
// TODO: Take Eigen vectors of means and variances

class MutualInformation {
public:
    MutualInformation(Copula & copula, std::vector<Point> & data, int min_pop = 10) : copula(copula) {

        // KDTree tree(data, copula, min_pop); // Update me
    }

private:
    // KDTree tree;
    Copula copula;

};

// Mutual information with normally distributed marginals
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

// Mutual information with NB distributed marginals
double mutual_information_nb(double mean1, double conc1, double mean2, double conc2, std::vector<Point> & data, int min_pop = 10)
{
    auto p_x_func = [&](int x) -> double {
        return nb2_base(x, mean1, conc1);
    };
    auto p_y_func = [&](int y) -> double {
        return nb2_base(y, mean2, conc2);
    };

    KDTree tree(data, min_pop);

    return tree.compute_mutual_information(p_x_func, p_y_func);

}

// Mutual information with NB distributed marginals
double mutual_information_nb(double mean1, double conc1, double mean2, double conc2, Eigen::MatrixXd & data, int min_pop = 10)
{
    // Real value input - quantise first

    std::vector<Point> point_samples = convertSamplesToPoints(data);

    return mutual_information_nb(mean1, conc1, mean2, conc2, point_samples, min_pop);
}
