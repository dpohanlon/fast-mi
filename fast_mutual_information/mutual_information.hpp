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

    MutualInformation() {
        this->copula = new Copula();
    }

    // Destructor to prevent memory leaks
    ~MutualInformation() {
        delete this->copula;
    }

    // Fix how to set this up with copula, tree, etc
    MutualInformation(std::vector<Point> & data, int min_pop = 10) {

        this->copula = new Copula();
        this->setData(data, min_pop);

    }

    void setNormalCopula(double mean1, double std_dev1, double mean2, double std_dev2)
    {
        // std::cout << "Copula " << mean1 << " " << std_dev1 << " "  << mean2 << " "  << std_dev2 << std::endl;
        setNormalPMF(mean1, std_dev1, mean2, std_dev2);
        setNormalCDF(mean1, std_dev1, mean2, std_dev2);
        setNormalICDF(mean1, std_dev1, mean2, std_dev2);
    }

    void setUniformCopula()
    {
        this->copula->p_x = [](double x) -> double { return 1.0; };
        this->copula->p_y = [](double y) -> double { return 1.0; };
        this->copula->cdf_x = [](double x) -> double { return x; };
        this->copula->cdf_y = [](double y) -> double { return y; };
        this->copula->icdf_x = [](double u) -> double { return u; };
        this->copula->icdf_y = [](double v) -> double { return v; };
    }

    void setData(std::vector<Point> & data, int min_pop = 10)
    {
        this->tree = KDTree(data, this->copula, min_pop);
    }

    void setNormalPMF(double mean1, double std_dev1, double mean2, double std_dev2)
    {
        this->copula->p_x = [=](double x) -> double { return normal_pdf(x, mean1, std_dev1); };
        this->copula->p_y = [=](double y) -> double { return normal_pdf(y, mean2, std_dev2); };
    }

    void setNormalCDF(double mean1, double std_dev1, double mean2, double std_dev2)
    {
        this->copula->cdf_x = [=](double x) -> double { return normal_cdf(x, mean1, std_dev1); };
        this->copula->cdf_y = [=](double y) -> double { return normal_cdf(y, mean2, std_dev2); };
    }

    void setNormalICDF( double mean1, double std_dev1, double mean2, double std_dev2)
    {
        this->copula->icdf_x = [=](double x) -> double { return normal_icdf(x, mean1, std_dev1); };
        this->copula->icdf_y = [=](double y) -> double { return normal_icdf(y, mean2, std_dev2); };
    }

    double mutual_information()
    {
        // TODO: Check if copula funcs are not null (and that we are set up)

        return tree.compute_mutual_information();

    }

private:
    KDTree tree;
    Copula * copula;

};

// Mutual information with normally distributed marginals
double mutual_information_normal(double mean1, double std_dev1, double mean2, double std_dev2, std::vector<Point> & data, int min_pop = 10)
{
    MutualInformation mi(data, min_pop);
    mi.setUniformCopula();

    return mi.mutual_information();

}

double mutual_information_normal(std::vector<Point> & data, int min_pop = 50)
{
    MutualInformation mi(data, min_pop);
    mi.setUniformCopula();

    return mi.mutual_information();

}

double mutual_information_normal(double mean1, double std_dev1, double mean2, double std_dev2, Eigen::MatrixXd & data, int min_pop = 10)
{
    // Real value input - quantise first

    std::vector<Point> point_samples = convertSamplesToPoints(data);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2, point_samples, min_pop);
}

// // Mutual information with NB distributed marginals
// double mutual_information_nb(double mean1, double conc1, double mean2, double conc2, std::vector<Point> & data, int min_pop = 10)
// {
//     auto p_x_func = [&](int x) -> double {
//         return nb2_base(x, mean1, conc1);
//     };
//     auto p_y_func = [&](int y) -> double {
//         return nb2_base(y, mean2, conc2);
//     };

//     KDTree tree(data, min_pop);

//     return tree.compute_mutual_information(p_x_func, p_y_func);

// }

// // Mutual information with NB distributed marginals
// double mutual_information_nb(double mean1, double conc1, double mean2, double conc2, Eigen::MatrixXd & data, int min_pop = 10)
// {
//     // Real value input - quantise first

//     std::vector<Point> point_samples = convertSamplesToPoints(data);

//     return mutual_information_nb(mean1, conc1, mean2, conc2, point_samples, min_pop);
// }
