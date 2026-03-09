#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <functional>
#include <unordered_map>
#include <Eigen/Sparse>
#include <cstdint>
#include <random>

#include "copula.hpp"
#include "fast_negative_binomial/fast_nb.hpp"
#include "kdtree.hpp"
#include "mvn.hpp"
#include "point.hpp"
#include "utils.hpp"

// Set up with a class, configure, then run MI calculation
// TODO: Take Eigen vectors of means and variances
//       This is also quite a chunky file, maybe split it up.

void init_parallel()
{
    omp_set_dynamic(0);

    Eigen::initParallel();
    Eigen::setNbThreads(1);
}

inline void make_twofold_split_indices(int n, uint64_t seed, Eigen::VectorXi& idxA, Eigen::VectorXi& idxB) {
    Eigen::VectorXi perm(n);
    for (int i = 0; i < n; ++i) perm(i) = i;

    std::mt19937_64 rng(seed);
    std::shuffle(perm.data(), perm.data() + n, rng);

    const int nA = n / 2;
    idxA = perm.head(nA);
    idxB = perm.tail(n - nA);
}

template<typename Vec>
struct IndexedPointIter {
    using Scalar = typename Vec::Scalar;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Point<Scalar>;
    using difference_type = std::ptrdiff_t;
    using reference = value_type;

    const Vec* x = nullptr;
    const Vec* y = nullptr;
    const int* idx = nullptr;
    int pos = 0;

    value_type operator*() const {
        const int i = idx[pos];
        return value_type{(*x)(i), (*y)(i)};
    }

    IndexedPointIter& operator++() { ++pos; return *this; }
    IndexedPointIter& operator--() { --pos; return *this; }

    IndexedPointIter operator+(difference_type d) const { return {x, y, idx, pos + static_cast<int>(d)}; }
    IndexedPointIter operator-(difference_type d) const { return {x, y, idx, pos - static_cast<int>(d)}; }
    difference_type operator-(const IndexedPointIter& o) const { return static_cast<difference_type>(pos - o.pos); }

    bool operator==(const IndexedPointIter& o) const { return pos == o.pos; }
    bool operator!=(const IndexedPointIter& o) const { return pos != o.pos; }
    bool operator<(const IndexedPointIter& o) const { return pos < o.pos; }
};

template<typename Vec>
struct IndexedPointRange {
    const Vec* x = nullptr;
    const Vec* y = nullptr;
    const Eigen::VectorXi* idx = nullptr;

    auto begin() const { return IndexedPointIter<Vec>{x, y, idx->data(), 0}; }
    auto end() const {
        return IndexedPointIter<Vec>{x, y, idx->data(), static_cast<int>(idx->size())};
    }
    int size() const { return idx->size(); }
};

template<typename Vec>
inline std::pair<double, int> crossfit_chi2_twofold(
    const Vec& x,
    const Vec& y,
    Copula<typename Vec::Scalar>* copula,
    int min_pop,
    const Eigen::VectorXi& idxA,
    const Eigen::VectorXi& idxB,
    double min_expected = 5.0
) {
    IndexedPointRange<Vec> A{&x, &y, &idxA};
    IndexedPointRange<Vec> B{&x, &y, &idxB};

    KDTree<typename Vec::Scalar> treeA(A.begin(), A.end(), copula, min_pop);
    auto [chi2_ab, df_ab] = treeA.chi2_holdout(B.begin(), B.end(), B.size(), min_expected);

    KDTree<typename Vec::Scalar> treeB(B.begin(), B.end(), copula, min_pop);
    auto [chi2_ba, df_ba] = treeB.chi2_holdout(A.begin(), A.end(), A.size(), min_expected);

    // std::cout << chi2_ab << " " <<  chi2_ba << std::endl;
    // std::cout << df_ab << " " <<  df_ba << std::endl;

    return {chi2_ab + chi2_ba, df_ab + df_ba};
}

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
    MutualInformation(const PointView<ColVec>& data, int min_pop = 10, bool zi = false)
      : zi(zi)
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
            return normal_pmf_discrete(x, mean1, std_dev1);
        };
        this->copula->p_y = [mean2, std_dev2](double y) -> double {
            return normal_pmf_discrete(y, mean2, std_dev2);
        };
    }

    void setNormalCDF(double mean1, double std_dev1, double mean2,
                      double std_dev2) {
        this->copula->cdf_x = [mean1, std_dev1](double x) -> double {
            return normal_cdf_discrete(x, mean1, std_dev1);
        };
        this->copula->cdf_y = [mean2, std_dev2](double y) -> double {
            return normal_cdf_discrete(y, mean2, std_dev2);
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

    void set_exposure(const Eigen::VectorXd& exposure) {
        exposure_vec = exposure;
        use_offsets = true;
    }

    void clear_exposure() {
        exposure_vec.resize(0);
        use_offsets = false;
    }

    void dumpTreeToCSV(const std::string& filename) const {
        tree.dumpSplittingValuesToCSV(filename);
    }

    Copula<T>* copula;

   private:
    KDTree<T> tree;
    bool zi = false;
    Eigen::VectorXd exposure_vec;
    bool use_offsets = false;

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

// Build F_mix[0..Kmax] where F_mix(k) = mean_j F_NB(k; mu0*e_j, r), use for exposure/offset correction
// counts_col is used only to get Kmax cheaply (max count in this feature).
// Averaging over cells: F_mix(k) = mean_j F_NB(k; mu0 * exposure[j], r)
inline std::vector<double> build_nb_mixture_cdf_lookup(
    const Eigen::VectorXi& counts_col,
    double mu0,
    double r,
    const Eigen::VectorXd& exposure
) {
    const int n = static_cast<int>(counts_col.size());
    const int Kmax = counts_col.maxCoeff();

    std::vector<double> F(Kmax + 1);
    Eigen::VectorXi kvec(n);

    for (int k = 0; k <= Kmax; ++k) {
        kvec.setConstant(k);
        // uses the exposure-aware, block-parallel, no-sort vector CDF
        Eigen::VectorXd cdf_k = nb2_cdf_vec_eigen_exposure(kvec, mu0, r, exposure);
        F[k] = cdf_k.mean();
    }
    return F;
}

// ZINB mixture: F_mix(k) = mean_j [ alpha + (1-alpha) * F_NB(k; mu0 * exposure[j], r) ]
// Not currently plumbed in!
inline std::vector<double> build_zinb_mixture_cdf_lookup(
    const Eigen::VectorXi& counts_col,
    double mu0,
    double r,
    double alpha,
    const Eigen::VectorXd& exposure
) {
    const int n = static_cast<int>(counts_col.size());
    const int Kmax = counts_col.maxCoeff();

    std::vector<double> F(Kmax + 1);
    Eigen::VectorXi kvec(n);

    for (int k = 0; k <= Kmax; ++k) {
        kvec.setConstant(k);
        Eigen::VectorXd cdf_k = zinb2_cdf_vec_eigen_exposure(kvec, mu0, r, alpha, exposure);
        F[k] = cdf_k.mean();
    }
    return F;
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

    auto cdf_x = [=](int x) -> double { return (x < 0) ? 0.0 : nb2_cdf_single(x, mean1, conc1); };
    auto cdf_y = [=](int y) -> double { return (y < 0) ? 0.0 : nb2_cdf_single(y, mean2, conc2); };

    MutualInformation<int> mi(data, min_pop);
    mi.setCDF(cdf_x, cdf_y);

    return mi.mutual_information();
}

// Without RLE, 3
std::pair<double, double> mutual_information_zinb(double mean1, double conc1, double alpha1, double mean2,
                             double conc2, double alpha2, std::vector<Point<int>>& data,
                             int min_pop = 25) {
    auto cdf_x = [=](int x) -> double { return (x < 0) ? 0.0 : zinb2_cdf_single(x, mean1, conc1, alpha1); };
    auto cdf_y = [=](int y) -> double { return (y < 0) ? 0.0 : zinb2_cdf_single(y, mean2, conc2, alpha2); };

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

int
mutual_information_zinb_dump_first_tree(Eigen::MatrixXi& samples,
                                       Eigen::VectorXd means,
                                       Eigen::VectorXd concs,
                                       Eigen::VectorXd alphas,
                                       const std::string& csv_filename,
                                       int min_pop = 25) {
    const int F = samples.cols();

    // Compute (0,1) sequentially so we can dump its tree once.
    if (F >= 2) {
        PointView pv01(samples.col(0), samples.col(1));
        MutualInformation<int> mi01(pv01, min_pop, true);

        auto cdf_x = [=](int x) -> double {
            return (x < 0) ? 0.0 : zinb2_cdf_single(x, means(0), concs(0), alphas(0));
        };
        auto cdf_y = [=](int y) -> double {
            return (y < 0) ? 0.0 : zinb2_cdf_single(y, means(1), concs(1), alphas(1));
        };

        mi01.setCDF(cdf_x, cdf_y);

        (void)mi01.mutual_information();
        mi01.dumpTreeToCSV(csv_filename);
    }

    return 0;
}


// Mutual information with NB distributed marginals, RLE
std::pair<double, double> mutual_information_nb(double mean1, double conc1, double mean2,
                             double conc2,
                             std::vector<std::pair<Point<int>, int>>& data,
                             int nPoints, Bounds<int> bounds,
                             int min_pop = 10) {

    auto cdf_x = [=](int x) -> double { return (x < 0) ? 0.0 : nb2_cdf_single(x, mean1, conc1); };
    auto cdf_y = [=](int y) -> double { return (y < 0) ? 0.0 : nb2_cdf_single(y, mean2, conc2); };

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

    init_parallel();

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

#pragma omp parallel for schedule(dynamic,1)
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

// #pragma omp parallel for schedule(dynamic,1)
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

    init_parallel();

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

#pragma omp parallel for schedule(dynamic,1)
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

    init_parallel();

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

#pragma omp parallel for schedule(dynamic,1)
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

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_nb(
const Eigen::Ref<const Eigen::MatrixXi>& samples,
const Eigen::Ref<const Eigen::VectorXd>& means,
const Eigen::Ref<const Eigen::VectorXd>& concs,
int min_pop = 25) {

    init_parallel();

    const int F = samples.cols();

    Eigen::MatrixXd mi(F, F);
    Eigen::MatrixXd chi2(F, F);

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();

    // std::cout << "Calculating mutual information for negative binomial distribution..." << std::endl;

#pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < samples.cols(); i++) {
        Eigen::VectorXi f1 = samples.col(i);
        for (int j = i + 1; j < samples.cols(); j++) {
            Eigen::VectorXi f2 = samples.col(j);

            auto [mi_ij, chi2_ij] = mutual_information_nb(means(i), concs(i), means(j), concs(j), f1, f2, min_pop);

            mi(i, j) = mi_ij;
            chi2(i, j) = chi2_ij;

        }
    }

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();

    return {mi, chi2};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_zinb(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    const Eigen::Ref<const Eigen::VectorXd>& means,
    const Eigen::Ref<const Eigen::VectorXd>& concs,
    const Eigen::Ref<const Eigen::VectorXd>& alphas,
int min_pop = 25) {

    init_parallel();

    const int F = samples.cols();
    Eigen::MatrixXd mi(F, F);
    Eigen::MatrixXd chi2(F, F);

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();

#pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {

            Eigen::VectorXi f1 = samples.col(i);
            Eigen::VectorXi f2 = samples.col(j);

            auto [mi_ij, chi2_ij] = mutual_information_zinb(means(i), concs(i), alphas(i), means(j), concs(j), alphas(j), f1, f2, min_pop);

            mi(i, j) = mi_ij;
            chi2(i, j) = chi2_ij;

        }
    }

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();

    return {mi, chi2};
}

Eigen::MatrixXd mutual_information_ml(Eigen::MatrixXi& samples) {

    init_parallel();

    int ncols = samples.cols();

    Eigen::MatrixXd results = Eigen::MatrixXd::Zero(ncols, ncols);

    #pragma omp parallel for schedule(dynamic,1)
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

    init_parallel();

    Eigen::MatrixXd results(samples.cols(), samples.cols());

#pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < samples.cols(); i++) {
        for (int j = i + 1; j < samples.cols(); j++) {

            Eigen::VectorXi f1 = samples.col(i);
            Eigen::VectorXi f2 = samples.col(j);

            // Overloaded to take two Eigen::VectorXi, rather than a matrix
            results(i, j) = mutual_information_binarised(f1, f2);
        }
    }

    return results;
}

// NEW: pairwise NB with exposures (mu0 on unit exposure)
std::pair<double, double> mutual_information_nb(
    double mu0_1, double conc1,
    double mu0_2, double conc2,
    const Eigen::VectorXi& f1,
    const Eigen::VectorXi& f2,
    const Eigen::VectorXd& exposure,
    int min_pop = 25
) {

    std::vector<Point<int>> point_samples = convertSamplesToPoints(f1, f2);

    // Exposure uses a mixture CDF with different effective mean per observation

    // Precompute mixture CDF lookup tables for each axis
    const auto Fx = build_nb_mixture_cdf_lookup(f1, mu0_1, conc1, exposure);
    const auto Fy = build_nb_mixture_cdf_lookup(f2, mu0_2, conc2, exposure);

    // Lambdas map integer k -> mixture CDF via table (safe for k in [0..Kmax])
    auto cdf_x = [Fx](int k) -> double {
        if (k < 0) return 0.0;
        if (k < static_cast<int>(Fx.size())) return Fx[k];
        return 1.0; // robust tail clamp
    };
    auto cdf_y = [Fy](int k) -> double {
        if (k < 0) return 0.0;
        if (k < static_cast<int>(Fy.size())) return Fy[k];
        return 1.0;
    };

    MutualInformation<int> mi(point_samples, min_pop, /*zi=*/false);
    mi.setCDF(cdf_x, cdf_y);
    return mi.mutual_information();
}

// all-pairs NB with exposures; 'means' carries mu0 (baseline) per feature
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_nb(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    const Eigen::Ref<const Eigen::VectorXd>& means,
    const Eigen::Ref<const Eigen::VectorXd>& concs,
    const Eigen::Ref<const Eigen::VectorXd>& exposure,
    int min_pop = 25
) {

    init_parallel();

    const int F = samples.cols();
    Eigen::MatrixXd mi(F, F);
    Eigen::MatrixXd chi2(F, F);

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();

    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < F; ++i) {
        for (int j = i + 1; j < F; ++j) {
            const Eigen::VectorXi f1 = samples.col(i);
            const Eigen::VectorXi f2 = samples.col(j);

            auto result = mutual_information_nb(
                means(i), concs(i),
                means(j), concs(j),
                f1, f2, exposure, min_pop
            );
            mi(i, j)   = result.first;
            chi2(i, j) = result.second;
        }
    }

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();

    return {mi, chi2};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> mutual_information_nb_sparse(
    Eigen::SparseMatrix<int, Eigen::ColMajor>& samples,
    Eigen::VectorXd means,
    Eigen::VectorXd concs,
    int min_pop = 25) {

    const int F = samples.cols();
    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);

    #pragma omp parallel for schedule(dynamic,1)
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

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
mutual_information_normal_crossfit(
    Eigen::MatrixXi& samples,
    Eigen::VectorXd means,
    Eigen::VectorXd std_devs,
    int min_pop = 25,
    uint64_t seed = 0,
    double min_expected = 5.0
) {
    init_parallel();

    const int N = samples.rows();
    const int F = samples.cols();

    Eigen::VectorXi idxA, idxB;
    make_twofold_split_indices(N, seed, idxA, idxB);

    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd df   = Eigen::MatrixXd::Zero(F, F);

    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < F; i++) {
        for (int j = i + 1; j < F; j++) {
            PointView pv(samples.col(i), samples.col(j));
            MutualInformation<int> mi_full(pv, min_pop);
            mi_full.setNormalCopula(means(i), std_devs(i), means(j), std_devs(j));

            const auto [mi_ij, _] = mi_full.mutual_information();

            const auto [chi2_cv, df_cv] = crossfit_chi2_twofold(
                samples.col(i),
                samples.col(j),
                mi_full.copula,
                min_pop,
                idxA,
                idxB,
                min_expected
            );

            mi(i, j) = mi_ij;
            chi2(i, j) = chi2_cv;
            df(i, j) = static_cast<double>(df_cv);
        }
    }

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();
    df.triangularView<Eigen::Lower>().setZero();

    return {mi, chi2, df};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
mutual_information_nb_crossfit(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    const Eigen::Ref<const Eigen::VectorXd>& means,
    const Eigen::Ref<const Eigen::VectorXd>& concs,
    int min_pop = 25,
    std::uint64_t seed = 0,
    double min_expected = 5.0
) {
    init_parallel();

    const int N = samples.rows();
    const int F = samples.cols();

    Eigen::VectorXi idxA, idxB;
    make_twofold_split_indices(N, seed, idxA, idxB);

    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd df   = Eigen::MatrixXd::Zero(F, F);

    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < F; i++) {
        for (int j = i + 1; j < F; j++) {
            PointView pv(samples.col(i), samples.col(j));

            MutualInformation<int> mi_full(pv, min_pop);

            const double mean1 = means(i);
            const double conc1 = concs(i);
            const double mean2 = means(j);
            const double conc2 = concs(j);

            auto cdf_x = [=](int x) -> double {
                return (x < 0) ? 0.0 : nb2_cdf_single(x, mean1, conc1);
            };
            auto cdf_y = [=](int y) -> double {
                return (y < 0) ? 0.0 : nb2_cdf_single(y, mean2, conc2);
            };

            mi_full.setCDF(cdf_x, cdf_y);

            const auto [mi_ij, _chi2_internal] = mi_full.mutual_information();

            const auto [chi2_cv, df_cv] = crossfit_chi2_twofold(
                samples.col(i),
                samples.col(j),
                mi_full.copula,
                min_pop,
                idxA,
                idxB,
                min_expected
            );

            mi(i, j)   = mi_ij;
            chi2(i, j) = chi2_cv;
            df(i, j)   = static_cast<double>(df_cv);
        }
    }

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();
    df.triangularView<Eigen::Lower>().setZero();

    return {mi, chi2, df};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
mutual_information_zinb_crossfit(
    const Eigen::Ref<const Eigen::MatrixXi>& samples,
    const Eigen::Ref<const Eigen::VectorXd>& means,
    const Eigen::Ref<const Eigen::VectorXd>& concs,
    const Eigen::Ref<const Eigen::VectorXd>& alphas,
    int min_pop = 25,
    std::uint64_t seed = 0,
    double min_expected = 5.0
) {
    init_parallel();

    const int N = samples.rows();
    const int F = samples.cols();

    Eigen::VectorXi idxA, idxB;
    make_twofold_split_indices(N, seed, idxA, idxB);

    Eigen::MatrixXd mi   = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd chi2 = Eigen::MatrixXd::Zero(F, F);
    Eigen::MatrixXd df   = Eigen::MatrixXd::Zero(F, F);

    #pragma omp parallel for schedule(dynamic,1)
    for (int i = 0; i < F; i++) {
        for (int j = i + 1; j < F; j++) {
            PointView pv(samples.col(i), samples.col(j));

            MutualInformation<int> mi_full(pv, min_pop, /*zi=*/true);

            const double mean1 = means(i);
            const double conc1 = concs(i);
            const double alpha1 = alphas(i);

            const double mean2 = means(j);
            const double conc2 = concs(j);
            const double alpha2 = alphas(j);

            auto cdf_x = [=](int x) -> double {
                return (x < 0) ? 0.0 : zinb2_cdf_single(x, mean1, conc1, alpha1);
            };
            auto cdf_y = [=](int y) -> double {
                return (y < 0) ? 0.0 : zinb2_cdf_single(y, mean2, conc2, alpha2);
            };

            mi_full.setCDF(cdf_x, cdf_y);

            const auto [mi_ij, _chi2_internal] = mi_full.mutual_information();

            const auto [chi2_cv, df_cv] = crossfit_chi2_twofold(
                samples.col(i),
                samples.col(j),
                mi_full.copula,
                min_pop,
                idxA,
                idxB,
                min_expected
            );

            mi(i, j)   = mi_ij;
            chi2(i, j) = chi2_cv;
            df(i, j)   = static_cast<double>(df_cv);
        }
    }

    mi.triangularView<Eigen::Lower>().setZero();
    chi2.triangularView<Eigen::Lower>().setZero();
    df.triangularView<Eigen::Lower>().setZero();

    return {mi, chi2, df};
}
