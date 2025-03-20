#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <functional>
#include <unordered_map>

#include "copula.hpp"
#include "fast_negative_binomial/fast_nb.hpp"
#include "kdtree.hpp"
#include "mvn.hpp"
#include "utils.hpp"

// Set up with a class, configure, then run MI calculation
// TODO: Take Eigen vectors of means and variances
//       This is also quite a chunky file, maybe split it up.

template <typename T>
class MutualInformation {
   public:
    MutualInformation() { this->copula = new Copula<T>(); }

    ~MutualInformation() { delete this->copula; }

    MutualInformation(std::vector<Point<T>>& data, int min_pop = 10) {
        this->copula = new Copula<T>();
        this->setData(data, min_pop);
    }

    MutualInformation(std::vector<std::pair<Point<int>, int>>& data,
                      int nPoints, Bounds<int> bounds, int max_points_per_leaf);

    void setNormalCopula(double mean1, double std_dev1, double mean2,
                         double std_dev2) {
        this->copula->name = "NORMAL";
        setNormalPMF(mean1, std_dev1, mean2, std_dev2);
        setNormalCDF(mean1, std_dev1, mean2, std_dev2);
        setNormalICDF(mean1, std_dev1, mean2, std_dev2);
    }

    void setUniformCopula() {
        this->copula->name = "UNIFORM";
        this->copula->p_x = [](double x) -> double { return 1.0; };
        this->copula->p_y = [](double y) -> double { return 1.0; };
        this->copula->cdf_x = [](double x) -> double { return x; };
        this->copula->cdf_y = [](double y) -> double { return y; };
        this->copula->icdf_x = [](double u) -> double { return u; };
        this->copula->icdf_y = [](double v) -> double { return v; };
    }

    void setData(const std::vector<Point<T>>& data, int min_pop = 10) {
        this->tree = KDTree<T>(data, this->copula, min_pop);
    }

    void setData(std::vector<std::pair<Point<int>, int>>& data, int nPoints,
                 Bounds<int> bounds, int min_pop = 10) {
        // Use the RLE constructor
        this->tree = KDTree<T>(data, nPoints, bounds, this->copula, min_pop);
    }

    void setNormalPMF(double mean1, double std_dev1, double mean2,
                      double std_dev2) {
        this->copula->p_x = [mean1, std_dev1](double x) -> double {
            return normal_pdf(x, mean1, std_dev1);
        };
        this->copula->p_y = [mean2, std_dev2](double y) -> double {
            return normal_pdf(y, mean2, std_dev2);
        };
    }

    void setNormalCDF(double mean1, double std_dev1, double mean2,
                      double std_dev2) {
        this->copula->cdf_x = [mean1, std_dev1](double x) -> double {
            return normal_cdf(x, mean1, std_dev1);
        };
        this->copula->cdf_y = [mean2, std_dev2](double y) -> double {
            return normal_cdf(y, mean2, std_dev2);
        };
    }

    void setNormalICDF(double mean1, double std_dev1, double mean2,
                       double std_dev2) {
        this->copula->icdf_x = [mean1, std_dev1](double x) -> double {
            return normal_icdf(x, mean1, std_dev1);
        };
        this->copula->icdf_y = [mean2, std_dev2](double y) -> double {
            return normal_icdf(y, mean2, std_dev2);
        };
    }

    void setPDF(DistributionFunctions<T>::pdf_f pdf_x,
                DistributionFunctions<T>::pdf_f pdf_y) {
        this->copula->name = "CUSTOM";
        this->copula->pdf_x = pdf_x;
        this->copula->pdf_y = pdf_y;
    }

    void setCDF(DistributionFunctions<T>::cdf_f cdf_x,
                DistributionFunctions<T>::cdf_f cdf_y) {
        this->copula->name = "CUSTOM";
        this->copula->cdf_x = cdf_x;
        this->copula->cdf_y = cdf_y;
    }

    void setICDF(DistributionFunctions<T>::icdf_f icdf_x,
                 DistributionFunctions<T>::icdf_f icdf_y) {
        this->copula->name = "CUSTOM";
        this->copula->icdf_x = icdf_x;
        this->copula->icdf_y = icdf_y;
    }

    double mutual_information() {
        // TODO: Check if copula funcs are not null (and that we are set up)

        return tree.compute_mutual_information();
    }

    double mutual_information_ml(const std::vector<std::pair<int, int>>& rle_x,
                                 const std::vector<std::pair<int, int>>& rle_y);

    Copula<T>* copula;

   private:
    KDTree<T> tree;
    // Copula<T>* copula;
};

template <>
MutualInformation<int>::MutualInformation(
    std::vector<std::pair<Point<int>, int>>& data, int nPoints,
    Bounds<int> bounds, int max_points_per_leaf) {
    this->copula = new Copula<int>();
    this->setData(data, nPoints, bounds, max_points_per_leaf);
}

// Unqualified these correspond to uniform distributions, in both the raw and
// transformed spaces. Qualified, these have the kd-tree calculated in the raw
// space, but the MI calculated in the transformed space using the copula.

template <typename T>
double mutual_information(std::vector<Point<T>>& data, int min_pop = 25) {
    MutualInformation<T> mi(data, min_pop);
    mi.setUniformCopula();

    return mi.mutual_information();
}

// Implementation for MutualInformation::mutual_information_naive_rle
template <>
double MutualInformation<int>::mutual_information_ml(
    const std::vector<std::pair<int, int>>& rle_x,
    const std::vector<std::pair<int, int>>& rle_y) {
    // Compute total number of samples (assuming rle_x covers the full dataset)
    int total = 0;
    for (const auto& p : rle_x) {
        total += p.second;
    }
    if (total == 0) return 0.0;

    double mi = 0.0;
    size_t i = 0, j = 0;
    int remaining_x = (i < rle_x.size() ? rle_x[i].second : 0);
    int remaining_y = (j < rle_y.size() ? rle_y[j].second : 0);

    while (i < rle_x.size() && j < rle_y.size()) {
        // Overlap count is the minimum remaining counts in the current runs.
        int count = std::min(remaining_x, remaining_y);

        // Joint probability for the overlapping segment.
        double p_xy = static_cast<double>(count) / total;
        // Marginal probabilities from the full run segments.
        double p_x = static_cast<double>(rle_x[i].second) / total;
        double p_y = static_cast<double>(rle_y[j].second) / total;

        if (p_xy > 0 && p_x > 0 && p_y > 0) {
            mi += p_xy * std::log(p_xy / (p_x * p_y));
        }

        // Update remaining counts and advance pointers if a run is exhausted.
        remaining_x -= count;
        remaining_y -= count;
        if (remaining_x == 0) {
            i++;
            if (i < rle_x.size()) {
                remaining_x = rle_x[i].second;
            }
        }
        if (remaining_y == 0) {
            j++;
            if (j < rle_y.size()) {
                remaining_y = rle_y[j].second;
            }
        }
    }

    return mi;
}


// Pass normal parameters so the CDF can be calculated on the fly
template <typename T>
double mutual_information_normal(double mean1, double std_dev1, double mean2,
                                 double std_dev2, std::vector<Point<T>>& data,
                                 int min_pop = 25) {
    MutualInformation<T> mi(data, min_pop);
    mi.setNormalCopula(mean1, std_dev1, mean2, std_dev2);

    return mi.mutual_information();
}

// Where the duplicate counting has already been done (e.g., from RLE
// representations)
double mutual_information_normal(double mean1, double std_dev1, double mean2,
                                 double std_dev2,
                                 std::vector<std::pair<Point<int>, int>>& data,
                                 int nPoints, Bounds<int> bounds,
                                 int min_pop = 25) {
    MutualInformation<int> mi(data, nPoints, bounds, min_pop);
    mi.setNormalCopula(mean1, std_dev1, mean2, std_dev2);

    return mi.mutual_information();
}

double mutual_information_normal(Eigen::MatrixXd& data, int min_pop = 25) {
    std::vector<Point<double>> point_samples = convertSamplesToPoints(data);

    return mutual_information(point_samples, min_pop);
}

double mutual_information_normal(double mean1, double std_dev1, double mean2,
                                 double std_dev2, Eigen::MatrixXd& data,
                                 int min_pop = 25) {

    Eigen::VectorXd mean(2);
    mean << mean1, mean2;

    Eigen::VectorXd variance(2);
    variance << std_dev1 * std_dev1, std_dev2 * std_dev2;

    Eigen::VectorXd std_dev = variance.array().sqrt();

    Eigen::MatrixXd uniform_samples = transformToUniform(data, mean, std_dev);

    std::vector<Point<double>> point_samples =
        convertSamplesToPoints(uniform_samples);

    return mutual_information(point_samples, min_pop);
}

template <typename T>
double mutual_information_normal(double mean1, double std_dev1, double mean2,
                                 double std_dev2, const T& data1,
                                 const T& data2, int min_pop = 25) {

    Eigen::VectorXd uniform_samples1 = transformToUniform(data1, mean1, std_dev1);
    Eigen::VectorXd uniform_samples2 = transformToUniform(data2, mean2, std_dev2);

    // Convert the uniform samples to points.
    std::vector<Point<double>> point_samples =
        convertSamplesToPoints(uniform_samples1, uniform_samples2);

    return mutual_information(point_samples, min_pop);
}


// TODO: Find a better way to integrate all of these parameters, particularly
// the total number of points and the overall bounds
double mutual_information_quantised_rle(double mean1, double std_dev1,
                                        double mean2, double std_dev2,
                                        std::vector<std::pair<int, int>> rle1,
                                        std::vector<std::pair<int, int>> rle2,
                                        int nPoints, int min_pop = 25) {
    std::vector<std::pair<Point<int>, int>> point_samples =
        runLengthDecoding(rle1, rle2);

    // A little inefficient, as we can cache these per feature separately
    Bounds<int> bounds = get_bounds(point_samples);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2,
                                     point_samples, nPoints, bounds, min_pop);
}

double mutual_information_quantised(double mean1, double std_dev1, double mean2,
                                    double std_dev2, Eigen::MatrixXd& data,
                                    int min_pop = 25) {
    std::vector<Point<int>> point_samples =
        convertSamplesToPointsQuantised(data);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2,
                                     point_samples, min_pop);
}

double mutual_information_quantised(double mean1, double std_dev1, double mean2,
                                    double std_dev2, Eigen::VectorXd& data1,
                                    Eigen::VectorXd data2, int min_pop = 25) {
    std::vector<Point<int>> point_samples =
        convertSamplesToPointsQuantised(data1, data2);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2,
                                     point_samples, min_pop);
}

// Without RLE, 3
double mutual_information_nb(double mean1, double conc1, double mean2,
                             double conc2, std::vector<Point<int>>& data,
                             int min_pop = 25) {
    auto cdf_x = [=](int x) -> double { return nb2_cdf_single(x, mean1, conc1); };
    auto cdf_y = [=](int y) -> double { return nb2_cdf_single(y, mean2, conc2); };

    // for (auto& point : data) {
    //     std::cout << point.x << " " << point.y << std::endl;
    // }

    MutualInformation<int> mi(data, min_pop);
    mi.setCDF(cdf_x, cdf_y);

    // std::cout << mi.copula->cdf_x(data[0].x) << " " << mi.copula->cdf_y(data[0].y) << std::endl;

    // std::cout << mi.copula->cdf_x(mean1) << std::endl;

    return mi.mutual_information();
}

// 2
double mutual_information_nb(double mean1, double conc1, double mean2,
                             double conc2, Eigen::VectorXi& data1,
                             Eigen::VectorXi data2, int min_pop = 25) {

    std::vector<Point<int>> point_samples =
        convertSamplesToPoints(data1, data2);

    return mutual_information_nb(mean1, conc1, mean2, conc2, point_samples,
                                 min_pop);
}

// Mutual information with NB distributed marginals, RLE
double mutual_information_nb(double mean1, double conc1, double mean2,
                             double conc2,
                             std::vector<std::pair<Point<int>, int>>& data,
                             int nPoints, Bounds<int> bounds,
                             int min_pop = 10) {
    auto cdf_x = [=](int x) -> double { return nb2_cdf_single(x, mean1, conc1); };
    auto cdf_y = [=](int y) -> double { return nb2_cdf_single(y, mean2, conc2); };

    MutualInformation<int> mi(data, nPoints, bounds, min_pop);
    mi.setCDF(cdf_x, cdf_y);

    return mi.mutual_information();
}

double mutual_information_zinb(double mean1, double conc1, double mean2,
                             double conc2, double alpha1, double alpha2,
                             std::vector<std::pair<Point<int>, int>>& data,
                             int nPoints, Bounds<int> bounds,
                             int min_pop = 10) {
    auto cdf_x = [=](int x) -> double { return zinb2_base(x, mean1, conc1, alpha1); };
    auto cdf_y = [=](int y) -> double { return zinb2_base(y, mean2, conc2, alpha2); };

    MutualInformation<int> mi(data, nPoints, bounds, min_pop);
    mi.setCDF(cdf_x, cdf_y);

    return mi.mutual_information();
}

double mutual_information_nb_quantised_rle(
    double mean1, double conc1, double mean2, double conc2,
    std::vector<std::pair<int, int>> rle1,
    std::vector<std::pair<int, int>> rle2, int nPoints, int min_pop = 25) {
    std::vector<std::pair<Point<int>, int>> point_samples =
        runLengthDecoding(rle1, rle2);

    // A little inefficient, as we can cache these per feature separately
    Bounds<int> bounds = get_bounds(point_samples);

    return mutual_information_nb(mean1, conc1, mean2, conc2, point_samples,
                                 nPoints, bounds, min_pop);
}

double mutual_information_zinb_quantised_rle(
    double mean1, double conc1, double mean2, double conc2, double alpha1, double alpha2,
    std::vector<std::pair<int, int>> rle1,
    std::vector<std::pair<int, int>> rle2, int nPoints, int min_pop = 25) {
    std::vector<std::pair<Point<int>, int>> point_samples =
        runLengthDecoding(rle1, rle2);

    // A little inefficient, as we can cache these per feature separately
    Bounds<int> bounds = get_bounds(point_samples);

    return mutual_information_zinb(mean1, conc1, mean2, conc2, alpha1, alpha2, point_samples,
                                 nPoints, bounds, min_pop);
}

// Mutual information with NB distributed marginals
double mutual_information_nb(double mean1, double conc1, double mean2,
                             double conc2, Eigen::MatrixXd& data,
                             int min_pop = 10) {
    // Real value input - quantise first

    std::vector<Point<int>> point_samples =
        convertSamplesToPointsQuantised(data);

    return mutual_information_nb(mean1, conc1, mean2, conc2, point_samples,
                                 min_pop);
}

// TODO: Pass a struct of params rather that something explicit to clean this up

// mi_normal_q_rle in python
Eigen::MatrixXd mutual_information_rle(Eigen::MatrixXi& samples,
                                       Eigen::VectorXd means,
                                       Eigen::VectorXd std_devs,
                                       int min_pop = 25) {
    // Beware of types - integer matrix input

    std::vector<std::vector<std::pair<int, int>>> rle(samples.cols());

    Eigen::MatrixXd results(samples.cols(), samples.cols());

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        rle[i] = runLengthEncoding(samples.col(i));
    }

    // I don't *think* that the rle vectors are modified, but replace all of the
    // downstream functions with const versions, to be sure

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {
            double mi = mutual_information_quantised_rle(
                       means(i), std_devs(i), means(j), std_devs(j), rle[i],
                       rle[j], samples.rows(), min_pop);

            results(i, j) = mi;
        }
    }

    return results;
}

Eigen::MatrixXd mutual_information_nb_rle(Eigen::MatrixXi& samples,
                                          Eigen::VectorXd means,
                                          Eigen::VectorXd concentrations,
                                          int min_pop = 25) {
    // Beware of types - integer matrix input

    std::vector<std::vector<std::pair<int, int>>> rle(samples.cols());

    Eigen::MatrixXd results(samples.cols(), samples.cols());

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        rle[i] = runLengthEncoding(samples.col(i));
    }

    // I don't *think* that the rle vectors are modified, but replace all of the
    // downstream functions with const versions, to be sure

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {
            double mi = mutual_information_nb_quantised_rle(
                means(i), concentrations(i), means(j), concentrations(j),
                rle[i], rle[j], samples.rows(), min_pop);

            results(i, j) = mi;
        }
    }

    return results;
}

// Handle alpha as a parameter always?
Eigen::MatrixXd mutual_information_zinb_rle(Eigen::MatrixXi& samples,
                                          Eigen::VectorXd means,
                                          Eigen::VectorXd concentrations,
                                          Eigen::VectorXd alphas,
                                          int min_pop = 25) {
    // Beware of types - integer matrix input

    std::vector<std::vector<std::pair<int, int>>> rle(samples.cols());

    Eigen::MatrixXd results(samples.cols(), samples.cols());

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        rle[i] = runLengthEncoding(samples.col(i));
    }

    // I don't *think* that the rle vectors are modified, but replace all of the
    // downstream functions with const versions, to be sure

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {
            double mi = mutual_information_zinb_quantised_rle(
                means(i), concentrations(i), means(j), concentrations(j), alphas(i), alphas(j),
                rle[i], rle[j], samples.rows(), min_pop);

            results(i, j) = mi;
        }
    }

    return results;
}

// mi_normal in python
Eigen::MatrixXd mutual_information_normal(const Eigen::MatrixXd& samples,
                                            const Eigen::VectorXd& means,
                                            const Eigen::VectorXd& std_devs,
                                            int min_pop = 25) {
    Eigen::MatrixXd results(samples.cols(), samples.cols());

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {
            auto f1 = samples.col(i);
            auto f2 = samples.col(j);

            results(i, j) = mutual_information_normal(
                means(i), std_devs(i), means(j), std_devs(j), f1, f2, min_pop);
        }
    }

    return results;
}

// mi_normal_q in python
Eigen::MatrixXd mutual_information_normal(Eigen::MatrixXi& samples,
                                          Eigen::VectorXd means,
                                          Eigen::VectorXd std_devs,
                                          int min_pop = 25) {

    Eigen::MatrixXd results(samples.cols(), samples.cols());

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {
            // Cast to double so we can use the same point quantisation class,
            // even though these should be ints already
            Eigen::VectorXd f1 = samples.col(i).cast<double>();
            Eigen::VectorXd f2 = samples.col(j).cast<double>();

            results(i, j) = mutual_information_quantised(
                means(i), std_devs(i), means(j), std_devs(j), f1, f2, min_pop);
        }
    }

    return results;
}
// mi_negative_binomial in python, 1
Eigen::MatrixXd mutual_information_nb(Eigen::MatrixXi& samples,
                                                    Eigen::VectorXd means,
                                                    Eigen::VectorXd concs,
                                                    int min_pop = 25) {

    Eigen::MatrixXd results(samples.cols(), samples.cols());

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {

            Eigen::VectorXi f1 = samples.col(i);
            Eigen::VectorXi f2 = samples.col(j);

            results(i, j) = mutual_information_nb(means(i), concs(i), means(j), concs(j), f1, f2, min_pop);

        }
    }

    return results;
}

// double mutual_information_ml(Eigen::VectorXi& f1, Eigen::VectorXi& f2) {
double mutual_information_ml(std::vector<std::pair<int, int>> rle_x, std::vector<std::pair<int, int>> rle_y) {

    MutualInformation<int> mi;

    return mi.mutual_information_ml(rle_x, rle_y);

}

// Cache in a slightly different way than the others, maybe slower
Eigen::MatrixXd mutual_information_ml(Eigen::MatrixXi& samples) {
    int ncols = samples.cols();
    Eigen::MatrixXd results(ncols, ncols);

    // Precompute and cache the RLE vectors using a map with the col index as key.
    std::unordered_map<int, std::vector<std::pair<int, int>>> rleCache;
    for (int i = 0; i < ncols; i++) {
        Eigen::VectorXi f = samples.col(i);
        rleCache[i] = runLengthEncoding(f);
    }

    #pragma omp parallel for
    for (int i = 0; i < ncols; i++) {
        for (int j = i + 1; j < ncols; j++) {
            results(i, j) = mutual_information_ml(rleCache[i], rleCache[j]);
        }
    }

    return results;
}
