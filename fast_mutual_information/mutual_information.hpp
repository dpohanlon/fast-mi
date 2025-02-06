#pragma once

#include <functional>

#include <Eigen/Dense>
#include "fast_negative_binomial/fast_nb.hpp"

#include "utils.hpp"
#include "mvn.hpp"
#include "kdtree.hpp"
#include "copula.hpp"

// Set up with a class, configure, then run MI calculation
// TODO: Take Eigen vectors of means and variances

// MI base class with two subclasses, one with straight data, one precalculated
// or do run length decoding inside?

template <typename T>
class MutualInformation {
public:

    MutualInformation() {
        this->copula = new Copula<T>();
    }

    ~MutualInformation() {
        delete this->copula;
    }

    MutualInformation(std::vector<Point<T>> & data, int min_pop = 10) {

        this->copula = new Copula<T>();
        this->setData(data, min_pop);

    }

    MutualInformation(std::vector<std::pair<Point<int>, int>>& data, int nPoints, Bounds<int> bounds, int max_points_per_leaf);

    // Strictly speaking, this is not necessary, but is nice for a comparison
    void setNormalCopula(double mean1, double std_dev1, double mean2, double std_dev2)
    {
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

    void setData(const std::vector<Point<T>> & data, int min_pop = 10)
    {
        this->tree = KDTree<T>(data, this->copula, min_pop);
    }

    void setData(std::vector<std::pair<Point<int>, int>> & data, int nPoints, Bounds<int> bounds, int min_pop = 10)
    {

        // Use the RLE constructor
        this->tree = KDTree<T>(data, nPoints, bounds, this->copula, min_pop);
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

    void setNormalICDF(double mean1, double std_dev1, double mean2, double std_dev2)
    {
        this->copula->icdf_x = [=](double x) -> double { return normal_icdf(x, mean1, std_dev1); };
        this->copula->icdf_y = [=](double y) -> double { return normal_icdf(y, mean2, std_dev2); };
    }

    void setPDF(DistributionFunctions<T>::pdf_f pdf_x, DistributionFunctions<T>::pdf_f pdf_y)
    {
        this->copula->pdf_x = pdf_x;
        this->copula->pdf_y = pdf_y;
    }

    void setCDF(DistributionFunctions<T>::cdf_f cdf_x, DistributionFunctions<T>::cdf_f cdf_y)
    {
        this->copula->cdf_x = cdf_x;
        this->copula->cdf_y = cdf_y;
    }

    void setICDF(DistributionFunctions<T>::icdf_f icdf_x, DistributionFunctions<T>::icdf_f icdf_y)
    {
        this->copula->icdf_x = icdf_x;
        this->copula->icdf_y = icdf_y;
    }

    double mutual_information()
    {
        // TODO: Check if copula funcs are not null (and that we are set up)

        return tree.compute_mutual_information();

    }

private:
    KDTree<T> tree;
    Copula<T> * copula;

};

template <>
MutualInformation<int>::MutualInformation(std::vector<std::pair<Point<int>, int>>& data, int nPoints, Bounds<int> bounds, int max_points_per_leaf)
{
    this->copula = new Copula<int>();
    this->setData(data, nPoints, bounds, max_points_per_leaf);
}

// Unqualified these correspond to uniform distributions, in both the raw and transformed spaces. Qualified, these have the kd-tree calculated in the raw space, but the MI calculated in the transformed space using the copula.

template <typename T>
double mutual_information(std::vector<Point<T>> & data, int min_pop = 25)
{
    MutualInformation<T> mi(data, min_pop);
    mi.setUniformCopula();

    return mi.mutual_information();

}

// Pass normal parameters so the CDF can be calculated on the fly
template <typename T>
double mutual_information_normal(double mean1, double std_dev1, double mean2, double std_dev2, std::vector<Point<T>> & data, int min_pop = 25)
{
    MutualInformation<T> mi(data, min_pop);
    mi.setNormalCDF(mean1, std_dev1, mean2, std_dev2);

    return mi.mutual_information();

}

// Where the duplicate counting has already been done (e.g., from RLE representations)
double mutual_information_normal(double mean1, double std_dev1, double mean2, double std_dev2, std::vector<std::pair<Point<int>, int>> & data, int nPoints, Bounds<int> bounds, int min_pop = 25)
{
    MutualInformation<int> mi(data, nPoints, bounds, min_pop);
    mi.setNormalCDF(mean1, std_dev1, mean2, std_dev2);

    return mi.mutual_information();

}

double mutual_information_normal(Eigen::MatrixXd & data, int min_pop = 25)
{
    std::vector<Point<double>> point_samples = convertSamplesToPoints(data);

    return mutual_information(point_samples, min_pop);
}

double mutual_information_normal(double mean1, double std_dev1, double mean2, double std_dev2, Eigen::MatrixXd & data, int min_pop = 25)
{

    Eigen::VectorXd mean(2);
    mean << mean1, mean2;

    Eigen::VectorXd variance(2);
    variance << std_dev1 * std_dev1, std_dev2 * std_dev2;

    Eigen::VectorXd std_dev = variance.array().sqrt();

    Eigen::MatrixXd uniform_samples = transformToUniform(data, mean, std_dev);

    std::vector<Point<double>> point_samples = convertSamplesToPoints(uniform_samples);

    return mutual_information(point_samples, min_pop);
}

// TODO: Find a better way to integrate all of these parameters, particularly the total number of points and the overall bounds
double mutual_information_quantised_rle(double mean1, double std_dev1, double mean2, double std_dev2, std::vector<std::pair<int, int>> rle1, std::vector<std::pair<int, int>> rle2, int nPoints, int min_pop = 25)
{

    std::vector<std::pair<Point<int>, int>> point_samples = runLengthDecoding(rle1, rle2);

    // A little inefficient, as we can cache these per feature separately
    Bounds<int> bounds = get_bounds(point_samples);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2, point_samples, nPoints, bounds, min_pop);
}

double mutual_information_quantised(double mean1, double std_dev1, double mean2, double std_dev2, Eigen::MatrixXd & data, int min_pop = 25)
{

    std::vector<Point<int>> point_samples = convertSamplesToPointsQuantised(data);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2, point_samples, min_pop);
}

// Mutual information with NB distributed marginals
double mutual_information_nb(double mean1, double conc1, double mean2, double conc2, std::vector<Point<int>> & data, int min_pop = 10)
{
    auto cdf_x = [=](double x) -> double { return nb2_base(x, mean1, conc1); };
    auto cdf_y = [=](double y) -> double { return nb2_base(y, mean2, conc2); };

    MutualInformation<int> mi(data, min_pop);
    mi.setCDF(cdf_x, cdf_y);

    return mi.mutual_information();

}

// Mutual information with NB distributed marginals
double mutual_information_nb(double mean1, double conc1, double mean2, double conc2, Eigen::MatrixXd & data, int min_pop = 10)
{
    // Real value input - quantise first

    std::vector<Point<int>> point_samples = convertSamplesToPointsQuantised(data);

    return mutual_information_nb(mean1, conc1, mean2, conc2, point_samples, min_pop);
}
