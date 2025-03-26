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

// Add this custom hash for std::pair (if not already defined)
struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

template <>
double MutualInformation<int>::mutual_information_ml(
    const std::vector<std::pair<int, int>>& rle_x,
    const std::vector<std::pair<int, int>>& rle_y) {

    int total = 0;
    for (const auto& p : rle_x) {
        total += p.second;
    }
    if (total == 0) return 0.0;

    std::unordered_map<int, int> marg_x, marg_y;
    for (const auto& p : rle_x) {
        marg_x[p.first] += p.second;
    }
    for (const auto& p : rle_y) {
        marg_y[p.first] += p.second;
    }

    std::unordered_map<std::pair<int, int>, int, pair_hash> joint_counts;
    size_t i = 0, j = 0;
    int remaining_x = (i < rle_x.size() ? rle_x[i].second : 0);
    int remaining_y = (j < rle_y.size() ? rle_y[j].second : 0);

    while (i < rle_x.size() && j < rle_y.size()) {
        int count = std::min(remaining_x, remaining_y);
        joint_counts[{rle_x[i].first, rle_y[j].first}] += count;

        remaining_x -= count;
        remaining_y -= count;
        if (remaining_x == 0) {
            i++;
            if (i < rle_x.size())
                remaining_x = rle_x[i].second;
        }
        if (remaining_y == 0) {
            j++;
            if (j < rle_y.size())
                remaining_y = rle_y[j].second;
        }
    }

    double mi = 0.0;
    for (const auto& kv : joint_counts) {
        int a = kv.first.first;
        int b = kv.first.second;
        int joint_count = kv.second;
        double p_xy = static_cast<double>(joint_count) / total;
        double p_x = static_cast<double>(marg_x[a]) / total;
        double p_y = static_cast<double>(marg_y[b]) / total;
        // Add a small constant to avoid log(0)
        mi += p_xy * std::log(p_xy / (p_x * p_y + 1e-16));
    }

    return mi;
}

// double mutual_information_ml(const Eigen::VectorXi &x, const Eigen::VectorXi &y) {
//     assert(x.size() == y.size() && "Input vectors must be of equal length");
//     const int n = x.size();
//     if(n == 0) return 0.0;

//     // Determine maximum values to allocate joint count matrix (assumes non-negative integers)
//     const int max_x = x.maxCoeff();
//     const int max_y = y.maxCoeff();

//     Eigen::MatrixXi joint = Eigen::MatrixXi::Zero(max_x + 1, max_y + 1);

//     for (int i = 0; i < n; ++i) {
//         joint(x(i), y(i))++;
//     }

//     // Compute marginal counts using Eigen vectorized operations
//     const Eigen::VectorXi rowSums = joint.rowwise().sum();
//     const Eigen::VectorXi colSums = joint.colwise().sum();

//     // Compute mutual information (using natural logarithm)
//     double MI = 0.0;
//     for (int i = 0; i <= max_x; ++i) {
//         for (int j = 0; j <= max_y; ++j) {
//             const int count = joint(i, j);
//             if (count > 0) {
//                 const double p_xy = static_cast<double>(count) / n;
//                 const double p_x  = static_cast<double>(rowSums(i)) / n;
//                 const double p_y  = static_cast<double>(colSums(j)) / n;
//                 MI += p_xy * std::log(p_xy / (p_x * p_y));
//             }
//         }
//     }

//     return MI;
// }

// New function: map-based mutual information calculation
// double mutual_information_ml(const Eigen::VectorXi &x, const Eigen::VectorXi &y) {
//     assert(x.size() == y.size() && "Input vectors must be of equal length");
//     const int n = x.size();
//     if(n == 0) return 0.0;

//     // Determine maximum value of y for key mapping
//     const int max_y = y.maxCoeff();

//     // Create joint distribution using an unordered_map, where each key is computed as:
//     // key = x(i) * (max_y + 1) + y(i)
//     std::unordered_map<int64_t, int> joint;
//     for (int i = 0; i < n; ++i) {
//         int64_t key = static_cast<int64_t>(x(i)) * (max_y + 1) + y(i);
//         joint[key]++;
//     }

//     // Compute marginal counts from the joint map
//     std::unordered_map<int, int> x_counts;
//     std::unordered_map<int, int> y_counts;
//     for (const auto &entry : joint) {
//         int xi = entry.first / (max_y + 1);
//         int yi = entry.first % (max_y + 1);
//         int count = entry.second;
//         x_counts[xi] += count;
//         y_counts[yi] += count;
//     }

//     // Calculate mutual information using natural logarithms
//     double MI = 0.0;
//     for (const auto &entry : joint) {
//         int xi = entry.first / (max_y + 1);
//         int yi = entry.first % (max_y + 1);
//         int count = entry.second;
//         double p_xy = static_cast<double>(count) / n;
//         double p_x = static_cast<double>(x_counts[xi]) / n;
//         double p_y = static_cast<double>(y_counts[yi]) / n;
//         MI += p_xy * std::log(p_xy / (p_x * p_y));
//     }
//     return MI;
// }

// Sparse
// Have a switch between this and the dense matrix? ~(max - min) / total?

// double mutual_information_ml(const Eigen::VectorXi &x, const Eigen::VectorXi &y) {
//     assert(x.size() == y.size() && "Input vectors must be of equal length");
//     const int n = x.size();
//     if(n == 0) return 0.0;

//     // Determine maximum values (assumes non-negative integers)
//     const int max_x = x.maxCoeff();
//     const int max_y = y.maxCoeff();

//     // Build triplet list for sparse matrix (each triplet represents an observed (x,y) pair)
//     std::vector<Eigen::Triplet<int>> triplets;
//     triplets.reserve(n);
//     for (int i = 0; i < n; ++i) {
//         triplets.emplace_back(x(i), y(i), 1);
//     }

//     // Create sparse matrix for joint distribution
//     Eigen::SparseMatrix<int> joint(max_x + 1, max_y + 1);
//     joint.setFromTriplets(triplets.begin(), triplets.end());

//     // Compute marginal counts using the nonzero entries of the sparse matrix
//     std::vector<int> x_counts(max_x + 1, 0);
//     std::vector<int> y_counts(max_y + 1, 0);
//     for (int k = 0; k < joint.outerSize(); ++k) {
//         for (Eigen::SparseMatrix<int>::InnerIterator it(joint, k); it; ++it) {
//             int i = it.row();
//             int j = it.col();
//             int count = it.value();
//             x_counts[i] += count;
//             y_counts[j] += count;
//         }
//     }

//     // Compute mutual information (using natural logarithms)
//     double MI = 0.0;
//     for (int k = 0; k < joint.outerSize(); ++k) {
//         for (Eigen::SparseMatrix<int>::InnerIterator it(joint, k); it; ++it) {
//             int i = it.row();
//             int j = it.col();
//             int count = it.value();
//             double p_xy = static_cast<double>(count) / n;
//             double p_x = static_cast<double>(x_counts[i]) / n;
//             double p_y = static_cast<double>(y_counts[j]) / n;
//             MI += p_xy * std::log(p_xy / (p_x * p_y));
//         }
//     }
//     return MI;
// }

// Function to quantize an Eigen::VectorXi to a uint8_t representation.
// This scales the input values to the range [0, 255] based on the maximum value in the vector.
Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> quantize_vector(const Eigen::VectorXi &v) {
    const int n = v.size();
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> vq(n);
    if (n == 0)
        return vq;

    int max_val = v.maxCoeff();
    if (max_val == 0) {
        vq.setZero();
        return vq;
    }

    // Scale each value to the range [0, 255] with rounding.
    for (int i = 0; i < n; ++i) {
        double scaled = 255.0 * v(i) / max_val;
        // Ensure the value is clamped within [0, 255].
        if (scaled < 0)
            scaled = 0;
        else if (scaled > 255)
            scaled = 255;
        vq(i) = static_cast<uint8_t>(std::round(scaled));
    }
    return vq;
}

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

// Function to compute mutual information using a dense matrix, operating on preprocessed uint8_t data.
// Additional optimizations for uint8_t data include:
//  - The joint distribution matrix is now at most 256 x 256, which improves cache locality.
//  - Loop unrolling or precomputed logarithm tables (not shown here) could be integrated,
//    since the number of bins is fixed and small.
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
// double mutual_information_ml(std::vector<std::pair<int, int>> rle_x, std::vector<std::pair<int, int>> rle_y) {

//     MutualInformation<int> mi;

//     return mi.mutual_information_ml(rle_x, rle_y);

// }

std::vector<int> spectral_ordering_nystrom(const Eigen::MatrixXi& samples, int target_dim, double sigma, int n_landmarks) {
    const int nrows = samples.rows();
    const int ncols = samples.cols();

    // Step 1: Create a random projection matrix R (ncols x target_dim)
    Eigen::MatrixXd R = Eigen::MatrixXd::NullaryExpr(ncols, target_dim, []() {
        return 2.0 * ((double) std::rand() / RAND_MAX) - 1.0;
    });

    // Step 2: Project the samples (cast to double) to get a (nrows x target_dim) matrix.
    Eigen::MatrixXd projected = samples.cast<double>() * R;

    // Step 3: Randomly select landmark indices.
    std::vector<int> all_indices(nrows);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::shuffle(all_indices.begin(), all_indices.end(), std::mt19937{std::random_device{}()});
    std::vector<int> landmarks(all_indices.begin(), all_indices.begin() + n_landmarks);

    // Step 4: Build W, the (n_landmarks x n_landmarks) similarity matrix among landmarks.
    Eigen::MatrixXd W(n_landmarks, n_landmarks);
    for (int i = 0; i < n_landmarks; ++i) {
        for (int j = i; j < n_landmarks; ++j) {
            double dist_sq = (projected.row(landmarks[i]) - projected.row(landmarks[j])).squaredNorm();
            double sim = std::exp(-dist_sq / (2 * sigma * sigma));
            W(i, j) = sim;
            W(j, i) = sim;
        }
    }

    // Step 5: Build C, the (nrows x n_landmarks) similarity matrix between all rows and landmarks.
    Eigen::MatrixXd C(nrows, n_landmarks);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < n_landmarks; ++j) {
            double dist_sq = (projected.row(i) - projected.row(landmarks[j])).squaredNorm();
            double sim = std::exp(-dist_sq / (2 * sigma * sigma));
            C(i, j) = sim;
        }
    }

    // Step 6: Compute the eigen decomposition of W.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(W);
    if (eigenSolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed on landmarks.");
    }
    Eigen::VectorXd evals = eigenSolver.eigenvalues();      // m x 1 vector
    Eigen::MatrixXd U = eigenSolver.eigenvectors();           // m x m matrix (each column is an eigenvector)

    // Step 7: Nystrom extension to approximate eigenvectors for the full nrows x nrows matrix.
    // Here, we compute V = C * U * Λ^{-1}, where Λ is the diagonal matrix of eigenvalues.
    // We need to be careful with small eigenvalues.
    Eigen::MatrixXd V = C * U;
    for (int i = 0; i < evals.size(); i++) {
        double lambda = evals(i);
        if (std::abs(lambda) > 1e-10) {
            V.col(i) /= lambda;
        }
    }

    // Step 8: Use the second column of V (corresponding approximately to the Fiedler vector)
    // as an ordering heuristic. (This assumes the eigenvalues are in ascending order.)
    if (V.cols() < 2) {
        throw std::runtime_error("Not enough eigenvectors computed.");
    }
    Eigen::VectorXd fiedler = V.col(1); // approximate Fiedler vector

    // Step 9: Create ordering of row indices based on the Fiedler vector.
    std::vector<int> order(nrows);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&fiedler](int a, int b) {
        return fiedler(a) < fiedler(b);
    });

    return order;
}

std::vector<int> greedy_ordering_using_projection(const Eigen::MatrixXi& samples, int target_dim) {
    int nrows = samples.rows();
    int ncols = samples.cols();

    // Create a random projection matrix (Gaussian)
    Eigen::MatrixXd R = Eigen::MatrixXd::Random(ncols, target_dim);
    // Project the integer samples into a lower-dimensional double space.
    Eigen::MatrixXd projected = samples.cast<double>() * R;

    // Greedy nearest neighbor ordering in the projected space.
    std::vector<bool> visited(nrows, false);
    std::vector<int> order;
    order.reserve(nrows);

    // Start with the first row (or choose one at random).
    order.push_back(0);
    visited[0] = true;

    for (int k = 1; k < nrows; ++k) {
        int last = order.back();
        int bestRow = -1;
        double bestDist = std::numeric_limits<double>::max();
        // Find the closest unvisited row (using Euclidean distance in projected space).
        for (int i = 0; i < nrows; ++i) {
            if (!visited[i]) {
                double dist = (projected.row(last) - projected.row(i)).norm();
                if (dist < bestDist) {
                    bestDist = dist;
                    bestRow = i;
                }
            }
        }
        order.push_back(bestRow);
        visited[bestRow] = true;
    }

    return order;
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

// // Cache in a slightly different way than the others, maybe slower
// Eigen::MatrixXd mutual_information_ml(Eigen::MatrixXi& samples) {
//     int ncols = samples.cols();

//     Eigen::MatrixXd results = Eigen::MatrixXd::Zero(ncols, ncols);

//     // int nrows = samples.rows();
//     // Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nrows, nrows);

//     // // Build the similarity matrix (using Hamming similarity as an example).
//     // for (int i = 0; i < nrows; ++i) {
//     //     for (int j = i; j < nrows; ++j) {
//     //         int sim = 0;
//     //         for (int k = 0; k < samples.cols(); ++k) {
//     //             if (samples(i, k) == samples(j, k)) sim++;
//     //         }
//     //         S(i, j) = sim;
//     //         S(j, i) = sim;  // symmetry
//     //     }
//     // }

//     // // Degree matrix D.
//     // Eigen::VectorXd d = S.rowwise().sum();
//     // Eigen::MatrixXd D = d.asDiagonal();

//     // // Laplacian L.
//     // Eigen::MatrixXd L = D - S;

//     // std::cout << "Computing decomposition" << std::endl;

//     // // Compute eigen-decomposition.
//     // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(L);
//     // Eigen::VectorXd fiedler = eigenSolver.eigenvectors().col(1); // second smallest eigenvector

//     // std::cout << "Computed decomposition" << std::endl;

//     // // Create an ordering by sorting the indices based on the Fiedler vector.
//     // std::vector<int> order(nrows);
//     // std::iota(order.begin(), order.end(), 0);
//     // std::sort(order.begin(), order.end(), [&fiedler](int a, int b) {
//     //     return fiedler[a] < fiedler[b];
//     // });

//     // Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(samples.rows());
//     // for (int i = 0; i < samples.rows(); i++) {
//     //     perm.indices()(i) = order[i];
//     // }

//     // // Apply the permutation to sort the rows of the matrix.
//     // Eigen::MatrixXi sorted_samples = perm * samples;

//     //

//     // // Compute a row key for each row (e.g., the sum of its elements).
//     // Eigen::VectorXi rowSums = samples.rowwise().sum();

//     // // Create an ordering vector.
//     // std::vector<int> order(samples.rows());
//     // std::iota(order.begin(), order.end(), 0);

//     // // Sort indices based on row sums.
//     // std::sort(order.begin(), order.end(), [&rowSums](int i, int j) {
//     //     return rowSums(i) < rowSums(j);
//     // });

//     // std::vector<int> order = greedy_ordering_using_projection(samples, 10);
//     std::vector<int> order = spectral_ordering_nystrom(samples, 50, 3, 100);

//     // Create a permutation matrix and apply it.
//     Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(samples.rows());
//     for (int i = 0; i < samples.rows(); i++) {
//         perm.indices()(i) = order[i];
//     }
//     Eigen::MatrixXi sorted_samples = perm * samples;

//     // Precompute and cache the RLE vectors using a map with the col index as key.
//     std::unordered_map<int, std::vector<std::pair<int, int>>> rleCache;
//     for (int i = 0; i < ncols; i++) {
//         Eigen::VectorXi f = sorted_samples.col(i);
//         rleCache[i] = runLengthEncoding(f);
//     }

//     #pragma omp parallel for
//     for (int i = 0; i < ncols; i++) {
//         for (int j = i + 1; j < ncols; j++) {
//             results(i, j) = mutual_information_ml(rleCache[i], rleCache[j]);
//         }
//     }

//     return results;
// }



// Eigen::MatrixXd mutual_information_nb_zinb(Eigen::MatrixXi& samples,
//                                                     Eigen::VectorXd means,
//                                                     Eigen::VectorXd concs,
//                                                     Eigen::VectorXd alphas,
//                                                     int min_pop = 25) {

//     Eigen::MatrixXd results(samples.cols(), samples.cols());

// #pragma omp parallel for
//     for (int i = 0; i < samples.cols(); i++) {
//         for (int j = i + 1; j < samples.cols(); j++) {

//             Eigen::VectorXi f1 = samples.col(i);
//             Eigen::VectorXi f2 = samples.col(j);

//             // Write me

//         }
//     }

//     return results;
// }

// Even with the 'simple' ML MI method, the agreement is weird
// Estimating the correlation is also weird, but super wide (naturally)
