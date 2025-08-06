#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <functional>
#include <unordered_map>
#include <Eigen/Sparse>
#include <cstdint>

#include "copula.hpp"
#include "fast_negative_binomial/fast_nb.hpp"
#include "kdtree.hpp"
#include "mvn.hpp"
#include "point.hpp"
#include "utils.hpp"

// Set up with a class, configure, then run MI calculation
// TODO: Take Eigen vectors of means and variances
//       This is also quite a chunky file, maybe split it up.

template <typename T>
class MutualInformation {
   public:
    MutualInformation() { this->copula = new Copula<T>(); }

    ~MutualInformation() { delete this->copula; }

    MutualInformation(std::vector<Point<T>>& data, int min_pop = 10,
                      bool zi = false) : zi(zi) {
        this->copula = new Copula<T>();
        this->setData(data, min_pop);
    }

    MutualInformation(std::vector<std::pair<Point<int>, int>>& data,
                      int nPoints, Bounds<int> bounds, int max_points_per_leaf, bool zi = false);

    template<typename ColVec>
    MutualInformation(const PointView<ColVec>& data, int min_pop = 10)
      : zi(false)
    {
        this->copula = new Copula<T>();
        setData(data, min_pop);
    }

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
        this->tree = KDTree<T>(data, this->copula, min_pop, this->zi);
    }

    // void setData(PointView data, int min_pop = 10) {
    //     this->tree = KDTree<T>(data.begin(), data.end(), this->copula, min_pop, this->zi);
    // }

    template<typename ColVec>
    void setData(const PointView<ColVec>& data, int min_pop = 10) {
        static_assert(std::is_same_v<typename ColVec::Scalar, T>,
                      "PointView's Scalar must match MutualInformation<T>::T");
        this->tree = KDTree<T>(data.begin(), data.end(),
                               this->copula, min_pop, this->zi);
    }

    void setData(std::vector<std::pair<Point<int>, int>>& data, int nPoints,
                 Bounds<int> bounds, int min_pop = 10) {
        // Use the RLE constructor
        this->tree = KDTree<T>(data, nPoints, bounds, this->copula, min_pop, this->zi);
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

    std::pair<double, double> mutual_information() {
        // TODO: Check if copula funcs are not null (and that we are set up)

        return tree.compute_mutual_information();
    }

    Copula<T>* copula;

   private:
    KDTree<T> tree;
    bool zi = false;

};

template <>
MutualInformation<int>::MutualInformation(
    std::vector<std::pair<Point<int>, int>>& data, int nPoints,
    Bounds<int> bounds, int max_points_per_leaf, bool zi) : zi(zi) {
    this->copula = new Copula<int>();
    this->setData(data, nPoints, bounds, max_points_per_leaf);
}

// Unqualified these correspond to uniform distributions, in both the raw and
// transformed spaces. Qualified, these have the kd-tree calculated in the raw
// space, but the MI calculated in the transformed space using the copula.

template <typename T>
std::pair<double, double> mutual_information(std::vector<Point<T>>& data, int min_pop = 25) {
    MutualInformation<T> mi(data, min_pop);
    mi.setUniformCopula();

    return mi.mutual_information();
}

// Add this custom hash for std::pair (if not already defined)
struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> quantize_vector_straight(const Eigen::VectorXi &v) {
    const int n = v.size();
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> vq(n);
    for (int i = 0; i < n; ++i) {
        vq(i) = static_cast<uint8_t>(v(i));
    }
    return vq;
}

std::vector<std::pair<int, int>> sort_pairs(const std::vector<int>& x, const std::vector<int>& y) {
    assert(x.size() == y.size() && "Input vectors must be of equal length");
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        pairs.emplace_back(x[i], y[i]);
    }
    // Sort pairs lexicographically: first by the first element, then by the second.
    std::sort(pairs.begin(), pairs.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return (a.first < b.first) || ((a.first == b.first) && (a.second < b.second));
    });
    return pairs;
}

std::vector<int> eigenVectorToStdVector(const Eigen::VectorXi &eigen_vec) {
    // Use the pointer to the first element and the pointer past the last element.
    return std::vector<int>(eigen_vec.data(), eigen_vec.data() + eigen_vec.size());
}

float mutual_information_ml(const Eigen::VectorXi &x, const Eigen::VectorXi &y) {
    assert(x.size() == y.size() && "Input vectors must be of equal length");
    const int n = x.size();
    if(n == 0) return 0.0;

    sort_pairs(eigenVectorToStdVector(x), eigenVectorToStdVector(y));

    // Quantize the input vectors using a straight type cast.
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> xq = quantize_vector_straight(x);
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> yq = quantize_vector_straight(y);

    // Determine the maximum values in the quantized vectors (should be <= 255).
    const int max_x = static_cast<int>(xq.maxCoeff());
    const int max_y = static_cast<int>(yq.maxCoeff());

    // Create a joint distribution matrix using uint8.
    // Note: This matrix's counts are stored as uint8 and can overflow if n > 255.
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> joint(max_x + 1, max_y + 1);
    joint.setZero();

    // Accumulate joint counts.
    for (int i = 0; i < n; ++i) {
        joint(static_cast<int>(xq(i)), static_cast<int>(yq(i)))++;
    }

    // Compute marginal counts. These are also computed as uint8.
    // Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> rowSums(max_x + 1);
    // Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> colSums(max_y + 1);
    // rowSums.setZero();
    // colSums.setZero();

    // for (int i = 0; i <= max_x; ++i) {
    //     for (int j = 0; j <= max_y; ++j) {
    //         rowSums(i) += joint(i, j);
    //     }
    // }
    // for (int j = 0; j <= max_y; ++j) {
    //     for (int i = 0; i <= max_x; ++i) {
    //         colSums(j) += joint(i, j);
    //     }
    // }

    // Compute the mutual information.
    double MI = 0.0;
    for (int i = 0; i <= max_x; ++i) {
        for (int j = 0; j <= max_y; ++j) {
            uint8_t count = joint(i, j);
            if (count > 0) {
                float p_xy = static_cast<float>(count) / n;
                float p_x  = 1.0; //static_cast<double>(rowSums(i)) / n;
                float p_y  = 1.0; //static_cast<double>(colSums(j)) / n;
                MI += p_xy * std::log(p_xy / (p_x * p_y));
            }
        }
    }
    return MI;
}


// Pass normal parameters so the CDF can be calculated on the fly
template <typename T>
std::pair<double, double> mutual_information_normal(double mean1, double std_dev1, double mean2,
                                 double std_dev2, std::vector<Point<T>>& data,
                                 int min_pop = 25) {
    MutualInformation<T> mi(data, min_pop);
    mi.setNormalCopula(mean1, std_dev1, mean2, std_dev2);

    return mi.mutual_information();
}

// Where the duplicate counting has already been done (e.g., from RLE
// representations)
std::pair<double, double> mutual_information_normal(double mean1, double std_dev1, double mean2,
                                 double std_dev2,
                                 std::vector<std::pair<Point<int>, int>>& data,
                                 int nPoints, Bounds<int> bounds,
                                 int min_pop = 25) {
    MutualInformation<int> mi(data, nPoints, bounds, min_pop);
    mi.setNormalCopula(mean1, std_dev1, mean2, std_dev2);

    return mi.mutual_information();
}

std::pair<double, double>  mutual_information_normal(Eigen::MatrixXd& data, int min_pop = 25) {
    std::vector<Point<double>> point_samples = convertSamplesToPoints(data);

    return mutual_information(point_samples, min_pop);
}

std::pair<double, double>  mutual_information_normal(double mean1, double std_dev1, double mean2,
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
std::pair<double, double>  mutual_information_normal(double mean1, double std_dev1, double mean2,
                                 double std_dev2, const T& data1,
                                 const T& data2, int min_pop = 25) {

    Eigen::VectorXd uniform_samples1 = transformToUniform(data1, mean1, std_dev1);
    Eigen::VectorXd uniform_samples2 = transformToUniform(data2, mean2, std_dev2);

    // Convert the uniform samples to points.
    std::vector<Point<double>> point_samples =
        convertSamplesToPoints(uniform_samples1, uniform_samples2);

    return mutual_information(point_samples, min_pop);
}


std::pair<double, double>  mutual_information_quantised(double mean1, double std_dev1, double mean2,
                                    double std_dev2, Eigen::MatrixXd& data,
                                    int min_pop = 25) {
    std::vector<Point<int>> point_samples =
        convertSamplesToPointsQuantised(data);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2,
                                     point_samples, min_pop);
}

std::pair<double, double>  mutual_information_quantised(double mean1, double std_dev1, double mean2,
                                    double std_dev2, Eigen::VectorXd& data1,
                                    Eigen::VectorXd data2, int min_pop = 25) {
    std::vector<Point<int>> point_samples =
        convertSamplesToPointsQuantised(data1, data2);

    return mutual_information_normal(mean1, std_dev1, mean2, std_dev2,
                                     point_samples, min_pop);
}

// Without RLE, 3
std::pair<double, double> mutual_information_nb(double mean1, double conc1, double mean2,
                             double conc2, std::vector<Point<int>>& data,
                             int min_pop = 25) {
    auto cdf_x = [=](int x) -> double { return nb2_cdf_single(x, mean1, conc1); };
    auto cdf_y = [=](int y) -> double { return nb2_cdf_single(y, mean2, conc2); };

    MutualInformation<int> mi(data, min_pop);
    mi.setCDF(cdf_x, cdf_y);

    return mi.mutual_information();
}

// Without RLE, 3
std::pair<double, double> mutual_information_zinb(double mean1, double conc1, double alpha1, double mean2,
                             double conc2, double alpha2, std::vector<Point<int>>& data,
                             int min_pop = 25) {
    auto cdf_x = [=](int x) -> double { return zinb2_cdf_single(x, mean1, conc1, alpha1); };
    auto cdf_y = [=](int y) -> double { return zinb2_cdf_single(y, mean2, conc2, alpha2); };

    MutualInformation<int> mi(data, min_pop, true);
    mi.setCDF(cdf_x, cdf_y);

    return mi.mutual_information();
}

// 2
std::pair<double, double>  mutual_information_nb(double mean1, double conc1, double mean2,
                             double conc2, Eigen::VectorXi& data1,
                             Eigen::VectorXi data2, int min_pop = 25) {

    std::vector<Point<int>> point_samples =
        convertSamplesToPoints(data1, data2);

    return mutual_information_nb(mean1, conc1, mean2, conc2, point_samples,
                                 min_pop);
}

std::pair<double, double>  mutual_information_zinb(double mean1, double conc1, double alpha1, double mean2,
                             double conc2, double alpha2, Eigen::VectorXi& data1,
                             Eigen::VectorXi data2, int min_pop = 25) {

    std::vector<Point<int>> point_samples =
        convertSamplesToPoints(data1, data2);

    return mutual_information_zinb(mean1, conc1, alpha1, mean2, conc2, alpha2, point_samples,
                                 min_pop);
}

// Mutual information with NB distributed marginals, RLE
std::pair<double, double> mutual_information_nb(double mean1, double conc1, double mean2,
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

// Mutual information with NB distributed marginals
std::pair<double, double> mutual_information_nb(double mean1, double conc1, double mean2,
                             double conc2, Eigen::MatrixXd& data,
                             int min_pop = 10) {
    // Real value input - quantise first

    std::vector<Point<int>> point_samples =
        convertSamplesToPointsQuantised(data);

    return mutual_information_nb(mean1, conc1, mean2, conc2, point_samples,
                                 min_pop);
}

// mi_normal in python
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_normal(const Eigen::MatrixXd& samples,
const Eigen::VectorXd& means,
const Eigen::VectorXd& std_devs,
int min_pop = 25) {

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {
            auto f1 = samples.col(i);
            auto f2 = samples.col(j);

            auto [mi_ij, chi2_ij] = mutual_information_normal(
                means(i), std_devs(i), means(j), std_devs(j), f1, f2, min_pop);
            mi(i, j) = mi_ij;
            chi2(i, j) = chi2_ij;
        }
    }

    return {mi, chi2};
}

// mi_normal_q in python
// std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_normal(Eigen::MatrixXi& samples,
//                                           Eigen::VectorXd means,
//                                           Eigen::VectorXd std_devs,
//                                           int min_pop = 25) {

//     Eigen::MatrixXd mi(samples.cols(), samples.cols());
//     Eigen::MatrixXd chi2(samples.cols(), samples.cols());

// #pragma omp parallel for
//     for (int i = 0; i < samples.cols(); i++) {
//         for (int j = i + 1; j < samples.cols(); j++) {
//             // Cast to double so we can use the same point quantisation class,
//             // even though these should be ints already
//             Eigen::VectorXd f1 = samples.col(i).cast<double>();
//             Eigen::VectorXd f2 = samples.col(j).cast<double>();

//             auto [mi_ij, chi2_ij] = mutual_information_quantised(
//                 means(i), std_devs(i), means(j), std_devs(j), f1, f2, min_pop);

//             mi(i, j) = mi_ij;
//             chi2(i, j) = chi2_ij;
//         }
//     }

//     return {mi, chi2};
// }

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_normal(Eigen::MatrixXi& samples,
Eigen::VectorXd means,
Eigen::VectorXd std_devs,
int min_pop = 25) {

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {

            PointView pv(samples.col(i), samples.col(j));

            MutualInformation<int> mutual_information(pv, min_pop);
            mutual_information.setNormalCopula(means(i), std_devs(i), means(j), std_devs(j));

            auto [mi_ij, chi2_ij] = mutual_information.mutual_information();

            mi(i, j) = mi_ij;
            chi2(i, j) = chi2_ij;
        }
    }

    return {mi, chi2};
}

// Sparse equivalent that avoids creating Point vectors
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_normal_sparse(const Eigen::SparseMatrix<int, Eigen::ColMajor>& samples,
Eigen::VectorXd means,
Eigen::VectorXd std_devs,
int min_pop = 25) {

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {

            auto [f1, f2] = reconstructDenseVectorsFromSparse(samples, i, j);
            PointView pv(f1, f2);

            MutualInformation<int> mutual_information(pv, min_pop);
            mutual_information.setNormalCopula(means(i), std_devs(i), means(j), std_devs(j));

            auto [mi_ij, chi2_ij] = mutual_information.mutual_information();

            mi(i, j)   = mi_ij;
            chi2(i, j) = chi2_ij;
        }
    }

    return {mi, chi2};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_nb(Eigen::MatrixXi& samples,
Eigen::VectorXd means,
Eigen::VectorXd concs,
int min_pop = 25) {

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {

            Eigen::VectorXi f1 = samples.col(i);
            Eigen::VectorXi f2 = samples.col(j);

            auto [mi_ij, chi2_ij] = mutual_information_nb(means(i), concs(i), means(j), concs(j), f1, f2, min_pop);

            mi(i, j) = mi_ij;
            chi2(i, j) = chi2_ij;

        }
    }

    return {mi, chi2};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_zinb(Eigen::MatrixXi& samples,
Eigen::VectorXd means,
Eigen::VectorXd concs,
Eigen::VectorXd alphas,
int min_pop = 25) {

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {

            Eigen::VectorXi f1 = samples.col(i);
            Eigen::VectorXi f2 = samples.col(j);

            auto [mi_ij, chi2_ij] = mutual_information_zinb(means(i), concs(i), alphas(i), means(j), concs(j), alphas(j), f1, f2, min_pop);

            mi(i, j) = mi_ij;
            chi2(i, j) = chi2_ij;

        }
    }

    return {mi, chi2};
}

Eigen::MatrixXd mutual_information_ml(Eigen::MatrixXi& samples) {
    int ncols = samples.cols();

    Eigen::MatrixXd results = Eigen::MatrixXd::Zero(ncols, ncols);

    #pragma omp parallel for
    for (int i = 0; i < ncols; i++) {
        for (int j = i + 1; j < ncols; j++) {
            results(i, j) = mutual_information_ml(samples.col(i), samples.col(j));
        }
    }

    return results;
}

double mutual_information_binarised(Eigen::VectorXi f1, Eigen::VectorXi f2) {

    double mi = 0.0;

    // Ensure the input features are binary and the same size
    if (f1.size() != f2.size()) {
        throw std::invalid_argument("Input features f1 and f2 are not the same size.");
    }

    int n = f1.size(); // Length of the vectors

    // Calculate marginal probabilities for f1 and f2, based on co-occurrences
    double f1_0 = 0, f1_1 = 0, f2_0 = 0, f2_1 = 0, p11 = 0, p00 = 0, p10 = 0, p01 = 0;

    for (int i = 0; i < n; i++) {
        f1_0 += (f1[i] == 0);
        f1_1 += (f1[i] == 1);
        f2_0 += (f2[i] == 0);
        f2_1 += (f2[i] == 1);

        if (f1[i] == 0 && f2[i] == 0) p00++;
        if (f1[i] == 0 && f2[i] == 1) p01++;
        if (f1[i] == 1 && f2[i] == 0) p10++;
        if (f1[i] == 1 && f2[i] == 1) p11++;
    }

    f1_0 /= n;
    f1_1 /= n;
    f2_0 /= n;
    f2_1 /= n;
    p00 /= n;
    p01 /= n;
    p10 /= n;
    p11 /= n;

    // Sum over each co-occurrence making sure only valid terms are calculated

    if (p00 > 0 && f1_0 > 0 && f2_0 > 0) mi += p00 * log(p00 / (f1_0 * f2_0));
    if (p01 > 0 && f1_0 > 0 && f2_1 > 0) mi += p01 * log(p01 / (f1_0 * f2_1));
    if (p10 > 0 && f1_1 > 0 && f2_0 > 0) mi += p10 * log(p10 / (f1_1 * f2_0));
    if (p11 > 0 && f1_1 > 0 && f2_1 > 0) mi += p11 * log(p11 / (f1_1 * f2_1));

    return mi;
}

Eigen::MatrixXd mutual_information_binarised(Eigen::MatrixXi& samples) {

    Eigen::MatrixXd results(samples.cols(), samples.cols());

#pragma omp parallel for
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {

            Eigen::VectorXi f1 = samples.col(i);
            Eigen::VectorXi f2 = samples.col(j);

            // This is overloaded to take two Eigen::VectorXi, rather than a matrix
            results(i, j) = mutual_information_binarised(f1, f2);
        }
    }

    return results;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_nb_sparse(
    Eigen::SparseMatrix<int, Eigen::ColMajor>& samples,
    Eigen::VectorXd means,
    Eigen::VectorXd concs,
    int min_pop = 25) {

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

    #pragma omp parallel for
    for (int i = 0; i < F; ++i) {
        for (int j = i + 1; j < F; ++j) {

            // Have to account for all zeros in MI
            auto [f1, f2] = reconstructDenseVectorsFromSparse(samples, i, j);

            auto [mij, c2ij] = mutual_information_nb(
                means(i), concs(i), means(j), concs(j),
                f1, f2,
                min_pop);

            mi(i, j)   = mij;
            chi2(i, j) = c2ij;
        }
    }

    return {mi, chi2};
}
