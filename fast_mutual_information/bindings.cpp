#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>

#include <type_traits>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <string>
#include <stdexcept>

#include "mutual_information.hpp"

// Looong looong matrix
typedef Eigen::Matrix<long int, Eigen::Dynamic, Eigen::Dynamic> MatrixXl;

namespace py = pybind11;

namespace {

// Jump buffer used for signal handling
static sigjmp_buf g_sigjmp_buf;

// Signal handler that longjmps back to the guarded context
void signal_handler(int signum) { siglongjmp(g_sigjmp_buf, signum); }

// Execute a callable while translating fatal signals into Python exceptions.
// Catches signals such as SIGSEGV and raises a RuntimeError instead of
// crashing the Python interpreter.
template <typename Func>
auto safe_execute(Func&& f) -> decltype(f()) {
    struct sigaction sa, old_segv, old_fpe, old_ill, old_bus;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGFPE, &sa, &old_fpe);
    sigaction(SIGILL, &sa, &old_ill);
    sigaction(SIGBUS, &sa, &old_bus);

    int sig = sigsetjmp(g_sigjmp_buf, 1);
    if (sig == 0) {
        auto result = f();
        sigaction(SIGSEGV, &old_segv, nullptr);
        sigaction(SIGFPE, &old_fpe, nullptr);
        sigaction(SIGILL, &old_ill, nullptr);
        sigaction(SIGBUS, &old_bus, nullptr);
        return result;
    }

    sigaction(SIGSEGV, &old_segv, nullptr);
    sigaction(SIGFPE, &old_fpe, nullptr);
    sigaction(SIGILL, &old_ill, nullptr);
    sigaction(SIGBUS, &old_bus, nullptr);
    throw std::runtime_error("Fatal signal (" + std::to_string(sig) +
                             ") detected during mutual information computation");
}

template <typename Derived>
void check_matrix_shape(const Eigen::DenseBase<Derived>& m, const char* name) {
    if (m.rows() == 0 || m.cols() == 0) {
        throw py::value_error(std::string(name) + " must have at least one row and one column");
    }
}

template <typename Derived>
void check_vector_size(const Eigen::DenseBase<Derived>& v, const char* name) {
    if (v.size() == 0) {
        throw py::value_error(std::string(name) + " must be non-empty");
    }
}

template <typename Derived>
void check_non_negative(const Eigen::DenseBase<Derived>& m, const char* name) {
    if ((m.derived().array() < typename Derived::Scalar(0)).any()) {
        throw py::value_error(std::string(name) + " must contain non-negative values");
    }
}

template <typename Derived>
void check_positive(const Eigen::DenseBase<Derived>& v, const char* name) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        if (!(v.derived().array().isFinite().all()) ||
            (v.derived().array() <= Scalar(0)).any()) {
            throw py::value_error(std::string(name) + " must contain positive, finite values");
        }
    } else {
        if ((v.derived().array() <= Scalar(0)).any()) {
            throw py::value_error(std::string(name) + " must contain positive values");
        }
    }
}

template <typename Derived>
void check_probabilities(const Eigen::DenseBase<Derived>& v, const char* name) {
    using Scalar = typename Derived::Scalar;
    static_assert(std::is_floating_point_v<Scalar>,
                  "check_probabilities expects a floating-point vector");
    if (!(v.derived().array().isFinite().all()) ||
        (v.derived().array() < Scalar(0)).any() ||
        (v.derived().array() > Scalar(1)).any()) {
        throw py::value_error(std::string(name) + " must contain values in the range [0, 1]");
    }
}

template <typename Derived>
void check_all_finite(const Eigen::DenseBase<Derived>& m, const char* name) {
    using Scalar = typename Derived::Scalar;
    if constexpr (std::is_floating_point_v<Scalar>) {
        if (!(m.derived().array().isFinite().all())) {
            throw py::value_error(std::string(name) + " contains NaN or Inf values");
        }
    }
}

void check_min_pop(int min_pop) {
    if (min_pop <= 0) {
        throw py::value_error("min_pop must be positive");
    }
}

void check_min_expected(double min_expected) {
    if (!std::isfinite(min_expected) || min_expected <= 0.0) {
        throw py::value_error("min_expected must be positive and finite");
    }
}

using IntMatrixArray =
    py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleMatrixArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;
using DoubleVectorArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

Eigen::MatrixXi as_matrix_xi(const IntMatrixArray& a, const char* name) {
    if (a.ndim() != 2) {
        throw py::value_error(std::string(name) + " must be a 2D array");
    }
    auto buf = a.unchecked<2>();
    Eigen::MatrixXi out(buf.shape(0), buf.shape(1));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        for (py::ssize_t j = 0; j < buf.shape(1); ++j) {
            out(i, j) = buf(i, j);
        }
    }
    return out;
}

Eigen::MatrixXd as_matrix_xd(const DoubleMatrixArray& a, const char* name) {
    if (a.ndim() != 2) {
        throw py::value_error(std::string(name) + " must be a 2D array");
    }
    auto buf = a.unchecked<2>();
    Eigen::MatrixXd out(buf.shape(0), buf.shape(1));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        for (py::ssize_t j = 0; j < buf.shape(1); ++j) {
            out(i, j) = buf(i, j);
        }
    }
    return out;
}

Eigen::VectorXd as_vector_xd(const DoubleVectorArray& a, const char* name) {
    if (a.ndim() != 1) {
        throw py::value_error(std::string(name) + " must be a 1D array");
    }
    auto buf = a.unchecked<1>();
    Eigen::VectorXd out(buf.shape(0));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        out(i) = buf(i);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(fast_mutual_information, m) {
    m.doc() = "Python bindings for fast mutual information computation.";

    m.def(
        "mi_ml",
        [](const IntMatrixArray& data) -> Eigen::MatrixXd {
            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            check_matrix_shape(data_i, "data");
            if (data_i.cols() < 2) {
                throw py::value_error("data must contain at least two columns");
            }
            check_non_negative(data_i, "data");
            return safe_execute([&] { return mutual_information_ml(data_i); });
        },
        py::arg("data"),
        "Fast mutual information using the naive discrete ML estimator.\n\n"
        "Parameters:\n"
        "    data (np.array): Data array of shape (Nsamples, Nfeatures).\n"
        "Returns:\n"
        "    float: The mutual information.");

    m.def(
        "mi_binarised",
        [](const IntMatrixArray& samples) -> Eigen::MatrixXd {
            Eigen::MatrixXi samples_i = as_matrix_xi(samples, "samples");
            check_matrix_shape(samples_i, "samples");
            if (samples_i.cols() < 2) {
                throw py::value_error(
                    "samples must contain at least two columns");
            }
            return safe_execute(
                [&] { return mutual_information_binarised(samples_i); });
        },
        py::arg("samples"),
        "Fast binarised mutual information computation.\n\n"
        "    data (np.array): Data array of shape (Nsamples, Nfeatures).\n"
        "Returns:\n"
        "    float: The mutual information.");

    m.def(
        "mi_normal",
        [](double mean1, double std_dev1, double mean2, double std_dev2,
           const DoubleMatrixArray& data, int min_pop) -> std::pair<double, double> {
            Eigen::MatrixXd data_d = as_matrix_xd(data, "data");
            check_matrix_shape(data_d, "data");
            check_all_finite(data_d, "data");
            if (data_d.cols() != 2) {
                throw py::value_error("data must have exactly two columns");
            }
            if (!std::isfinite(mean1) || !std::isfinite(mean2)) {
                throw py::value_error("means must be finite");
            }
            if (!std::isfinite(std_dev1) || std_dev1 <= 0 ||
                !std::isfinite(std_dev2) || std_dev2 <= 0) {
                throw py::value_error(
                    "std_dev1 and std_dev2 must be positive and finite");
            }
            check_min_pop(min_pop);
            return safe_execute([&] {
                return mutual_information_normal(mean1, std_dev1, mean2,
                                                 std_dev2, data_d, min_pop);
            });
        },
        py::arg("mean1"), py::arg("std_dev1"), py::arg("mean2"),
        py::arg("std_dev2"), py::arg("data"), py::arg("min_pop") = 25,
        "Fast mutual information computation with normally distributed "
        "marginals.\n\n"
        "Parameters:\n"
        "    mean1 (float): Mean of the first axis normal distribution.\n"
        "    std_dev1 (float): Standard-deviation of the first axis normal "
        "distribution.\n"
        "    mean2 (float): Mean of the second axis normal distribution\n"
        "    std_dev2 (float): Standard-deviation of the second axis normal "
        "distribution.\n"
        "    data (np.array): Data array of shape (Nsamples, 2).\n"
        "    min_pop (int): Mininmum bin population.\n\n"
        "Returns:\n"
        "    float: The mutual information.");

    m.def(
        "mi_normal",
        [](const DoubleMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& std_devs,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXd data_d = as_matrix_xd(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd std_devs_d = as_vector_xd(std_devs, "std_devs");
            check_matrix_shape(data_d, "data");
            check_vector_size(means_d, "means");
            check_vector_size(std_devs_d, "std_devs");
            check_all_finite(data_d, "data");
            check_all_finite(means_d, "means");
            check_all_finite(std_devs_d, "std_devs");
            if (data_d.cols() != means_d.size() || data_d.cols() != std_devs_d.size()) {
                throw py::value_error(
                    "data columns must match length of means and std_devs");
            }
            check_positive(std_devs_d, "std_devs");
            check_min_pop(min_pop);
            return safe_execute([&] {
                return mutual_information_normal(data_d, means_d, std_devs_d,
                                                 min_pop);
            });
        },
        py::arg("data"), py::arg("means"), py::arg("std_devs"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with normally distributed "
        "marginals.\n\n"
        "Parameters:\n"
        "    data (np.array): Integer data array of shape (Nsamples, "
        "Nfeatures).\n"
        "    means (np.array): Means of each normal distribution (Nfeatures, "
        "1).\n"
        "    std_devs (np.array): Standard deviations of each normal "
        "distribution "
        "(Nfeatures, 1).\n"
        "    min_pop (int): Mininmum bin population.\n\n"
        "Returns:\n"
        "    np.array: Array of mutual information values.");

    m.def(
        "mi_normal",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& variances,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd variances_d = as_vector_xd(variances, "variances");
            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(variances_d, "variances");
            if (data_i.cols() != means_d.size() ||
                data_i.cols() != variances_d.size()) {
                throw py::value_error(
                    "data columns must match length of means and variances");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(variances_d, "variances");
            check_positive(variances_d, "variances");
            check_min_pop(min_pop);
            return safe_execute([&] {
                return mutual_information_normal(data_i, means_d, variances_d,
                                                 min_pop);
            });
        },
        py::arg("data"), py::arg("means"), py::arg("variances"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with normally distributed "
        "marginals, quantised for integer inputs.\n\n"
        "Parameters:\n"
        "    data (np.array): Integer data array of shape (Nsamples, "
        "Nfeatures).\n"
        "    means (np.array): Means of each normal distribution (Nfeatures, "
        "1).\n"
        "    variances (np.array): Variances of each normal distribution "
        "(Nfeatures, 1).\n"
        "    min_pop (int): Mininmum bin population.\n\n"
        "Returns:\n"
        "    np.array: Array of mutual information values.");

    m.def(
        "mi_normal",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& variances,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd variances_d = as_vector_xd(variances, "variances");
            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(variances_d, "variances");
            if (data_i.cols() != means_d.size() ||
                data_i.cols() != variances_d.size()) {
                throw py::value_error(
                    "data columns must match length of means and variances");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(variances_d, "variances");
            check_positive(variances_d, "variances");
            check_min_pop(min_pop);
            return safe_execute([&] {
                return mutual_information_normal(data_i, means_d, variances_d,
                                                 min_pop);
            });
        },
        py::arg("data"), py::arg("means"), py::arg("variances"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with normally distributed "
        "marginals, quantised for integer inputs.\n\n"
        "Parameters:\n"
        "    data (np.array): Integer data array of shape (Nsamples, "
        "Nfeatures).\n"
        "    means (np.array): Means of each normal distribution (Nfeatures, "
        "1).\n"
        "    variances (np.array): Variances of each normal distribution "
        "(Nfeatures, 1).\n"
        "    min_pop (int): Mininmum bin population.\n\n"
        "Returns:\n"
        "    np.array: Array of mutual information values.");

    // Only the concentration formulation (i.e., 'nb2') for the moment

    m.def(
        "mi_negative_binomial",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd concentrations_d = as_vector_xd(concentrations, "concentrations");
            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(concentrations_d, "concentrations");
            if (data_i.cols() != means_d.size() ||
                data_i.cols() != concentrations_d.size()) {
                throw py::value_error(
                    "data columns must match length of means and "
                    "concentrations");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(concentrations_d, "concentrations");
            check_positive(means_d, "means");
            check_positive(concentrations_d, "concentrations");
            check_min_pop(min_pop);

            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_nb(data_i, means_d, concentrations_d,
                                             min_pop);
            });
        },
        py::arg("data"), py::arg("means"), py::arg("concentrations"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with negative binomial "
        "marginals.\n\n"
        "Parameters:\n"
        "    data (np.array): Integer data array of shape (Nsamples, "
        "Nfeatures).\n"
        "    means (np.array): Means of each negative binomial distribution "
        "(Nfeatures, 1).\n"
        "    concentrations (np.array): concentrations of each negative "
        "binomial distribution (Nfeatures, 1).\n"
        "    min_pop (int): Mininmum bin population.\n\n"
        "Returns:\n"
        "    np.array: Array of mutual information values.");

    m.def(
        "mi_zero_inflated_negative_binomial_dump_first_tree",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           const DoubleVectorArray& alphas,
           const std::string& csv_filename,
           int min_pop) -> int {
            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd concentrations_d = as_vector_xd(concentrations, "concentrations");
            Eigen::VectorXd alphas_d = as_vector_xd(alphas, "alphas");
            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(concentrations_d, "concentrations");
            check_vector_size(alphas_d, "alphas");
            if (data_i.cols() != means_d.size() ||
                data_i.cols() != concentrations_d.size() ||
                data_i.cols() != alphas_d.size()) {
                throw py::value_error(
                    "data columns must match length of means, concentrations, and alphas");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(concentrations_d, "concentrations");
            check_all_finite(alphas_d, "alphas");
            check_positive(means_d, "means");
            check_positive(concentrations_d, "concentrations");
            check_min_pop(min_pop);

            return safe_execute([&] {
                return mutual_information_zinb_dump_first_tree(
                    data_i, means_d, concentrations_d, alphas_d, csv_filename, min_pop);
            });
        },
        py::arg("data"), py::arg("means"), py::arg("concentrations"),
        py::arg("alphas"),
        py::arg("csv_filename"),
        py::arg("min_pop") = 25,
        "Compute zero-inflated negative-binomial mutual information internally and dump the "
        "KD-tree splits for the first feature pair (0,1) to a CSV file.\n\n"
        "Parameters:\n"
        "    data (np.array): Integer data array of shape (Nsamples, Nfeatures).\n"
        "    means (np.array): Means of each ZINB marginal (Nfeatures, 1).\n"
        "    concentrations (np.array): Concentrations of each ZINB marginal (Nfeatures, 1).\n"
        "    alphas (np.array): Zero-inflation parameters of each ZINB marginal (Nfeatures, 1).\n"
        "    csv_filename (str): Output path for the CSV dump.\n"
        "    min_pop (int): Minimum bin population.\n\n"
        "Returns:\n"
        "    int");

    m.def(
        "mi_negative_binomial_zi",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           const DoubleVectorArray& alphas,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd concentrations_d = as_vector_xd(concentrations, "concentrations");
            Eigen::VectorXd alphas_d = as_vector_xd(alphas, "alphas");
            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(concentrations_d, "concentrations");
            check_vector_size(alphas_d, "alphas");
            if (data_i.cols() != means_d.size() ||
                data_i.cols() != concentrations_d.size() ||
                data_i.cols() != alphas_d.size()) {
                throw py::value_error(
                    "data columns must match length of means, concentrations "
                    "and alphas");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(concentrations_d, "concentrations");
            check_all_finite(alphas_d, "alphas");
            check_positive(means_d, "means");
            check_positive(concentrations_d, "concentrations");
            check_probabilities(alphas_d, "alphas");
            check_min_pop(min_pop);
            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_zinb(data_i, means_d, concentrations_d,
                                               alphas_d, min_pop);
            });
        },
        py::arg("data"), py::arg("means"), py::arg("concentrations"),
        py::arg("alphas"), py::arg("min_pop") = 25,
        "Fast mutual information computation with negative binomial "
        "marginals.\n\n"
        "Parameters:\n"
        "    data (np.array): Integer data array of shape (Nsamples, "
        "(Nfeatures, 1).\n"
        "    means (np.array): Means of each negative binomial distribution "
        "(Nfeatures, 1).\n"
        "    concentrations (np.array): concentrations of each negative "
        "binomial distribution (Nfeatures, 1).\n"
        "    alphas (np.array): Zero inflation probabilities (alpha -> 0, pure "
        "NB) "
        "(Nfeatures, 1).\n"
        "    min_pop (int): Mininmum bin population.\n\n"
        "Returns:\n"
        "    np.array: Array of mutual information values.");

    m.def(
        "mi_negative_binomial_exposure",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           const DoubleVectorArray& exposure,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {

            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd concentrations_d = as_vector_xd(concentrations, "concentrations");
            Eigen::VectorXd exposure_d = as_vector_xd(exposure, "exposure");

            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(concentrations_d, "concentrations");
            check_vector_size(exposure_d, "exposure");
            if (data_i.cols() != means_d.size() ||
                data_i.cols() != concentrations_d.size()) {
                throw py::value_error(
                    "data columns must match length of means, concentrations");
            }
            if (data_i.rows() != exposure_d.size()) {
                throw py::value_error(
                    "data rows must match length of exposure");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(concentrations_d, "concentrations");
            check_all_finite(exposure_d, "exposure");
            check_positive(means_d, "means");
            check_positive(concentrations_d, "concentrations");
            check_positive(exposure_d, "exposure");
            check_min_pop(min_pop);

            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_nb(data_i, means_d, concentrations_d,
                                             exposure_d, min_pop);
            });

        },
        py::arg("data"),
        py::arg("means"),
        py::arg("concentrations"),
        py::arg("exposure"),
        py::arg("min_pop") = 25,
        "Fast MI with NB marginals **and per-sample exposure offsets**.\n"
        "Interpret 'means_mu0' as baseline means on unit exposure; the per-sample\n"
        "mean is mu_j = mu0 * exposure[j]. The marginal CDF used by the kd-tree is\n"
        "the mixture F_mix(k) = mean_j F_NB(k; mu0*exposure[j], r)."
    );

    // Sparse matrix versions

    m.def(
        "mi_normal_sparse",
        [](Eigen::SparseMatrix<int>& data, const DoubleVectorArray& means,
           const DoubleVectorArray& std_devs, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd std_devs_d = as_vector_xd(std_devs, "std_devs");
            return mutual_information_normal_sparse(data, means_d, std_devs_d, min_pop);
        },
        py::arg("data"), py::arg("means"), py::arg("std_devs"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with normally distributed "
        "marginals using sparse matrix input.\n\n"
        "Parameters:\n"
        "    data (scipy.sparse.csr_matrix): Sparse integer data array of shape (Nsamples, "
        "Nfeatures).\n"
        "    means (np.array): Means of each normal distribution (Nfeatures, "
        "1).\n"
        "    std_devs (np.array): Standard deviations of each normal distribution "
        "(Nfeatures, 1).\n"
        "    min_pop (int): Minimum bin population.\n\n"
        "Returns:\n"
        "    np.array: Array of mutual information values.");

    m.def(
        "mi_negative_binomial_sparse",
        [](Eigen::SparseMatrix<int>& data, const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd concentrations_d = as_vector_xd(concentrations, "concentrations");
            return mutual_information_nb_sparse(data, means_d, concentrations_d, min_pop);
        },
        py::arg("data"), py::arg("means"), py::arg("concentrations"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with negative binomial "
        "marginals using sparse matrix input (efficient conversion).\n\n"
        "Parameters:\n"
        "    data (scipy.sparse.csr_matrix): Sparse integer data array of shape (Nsamples, "
        "Nfeatures).\n"
        "    means (np.array): Means of each negative binomial distribution "
        "(Nfeatures, 1).\n"
        "    concentrations (np.array): Concentrations of each negative "
        "binomial distribution (Nfeatures, 1).\n"
        "    min_pop (int): Minimum bin population.\n\n"
        "Returns:\n"
        "    np.array: Array of mutual information values.");

    m.def(
        "mi_normal_crossfit",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& std_devs,
           int min_pop,
           std::uint64_t seed,
           double min_expected)
           -> std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd std_devs_d = as_vector_xd(std_devs, "std_devs");
            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(std_devs_d, "std_devs");
            if (data_i.cols() < 2) {
                throw py::value_error("data must contain at least two columns");
            }
            if (data_i.cols() != means_d.size() || data_i.cols() != std_devs_d.size()) {
                throw py::value_error("data columns must match length of means and std_devs");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(std_devs_d, "std_devs");
            check_positive(std_devs_d, "std_devs");
            check_min_pop(min_pop);
            check_min_expected(min_expected);

            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_normal_crossfit(
                    data_i, means_d, std_devs_d, min_pop, seed, min_expected
                );
            });
        },
        py::arg("data"),
        py::arg("means"),
        py::arg("std_devs"),
        py::arg("min_pop") = 25,
        py::arg("seed") = 0,
        py::arg("min_expected") = 5.0,
        "Cross-fit (2-fold) Pearson chi-square for the copula-independence null.\n\n"
        "This builds the KD-tree partition on fold A and evaluates chi-square on fold B,\n"
        "then swaps folds and sums the two chi-square values. The dof returned is the\n"
        "sum of fold-specific effective bin counts minus constraints (after merging bins\n"
        "with expected count < min_expected).\n\n"
        "Parameters:\n"
        "    data (np.array[int]): shape (Nsamples, Nfeatures)\n"
        "    means (np.array[float]): per-feature normal means\n"
        "    std_devs (np.array[float]): per-feature normal std devs\n"
        "    min_pop (int): minimum training-bin population during kd-tree construction\n"
        "    seed (int): RNG seed controlling the fold split\n"
        "    min_expected (float): minimum expected count per test bin (bins below are merged)\n\n"
        "Returns:\n"
        "    (mi, chi2_cv, dof_cv): each is (Nfeatures, Nfeatures) upper-triangular.\n"
    );

    m.def(
        "mi_normal_crossfit",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& std_devs,
           int min_pop,
           std::uint64_t seed,
           double min_expected)
           -> std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd std_devs_d = as_vector_xd(std_devs, "std_devs");
            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(std_devs_d, "std_devs");
            if (data_i.cols() < 2) {
                throw py::value_error("data must contain at least two columns");
            }
            if (data_i.cols() != means_d.size() || data_i.cols() != std_devs_d.size()) {
                throw py::value_error("data columns must match length of means and std_devs");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(std_devs_d, "std_devs");
            check_positive(std_devs_d, "std_devs");
            check_min_pop(min_pop);
            check_min_expected(min_expected);

            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_normal_crossfit(
                    data_i, means_d, std_devs_d, min_pop, seed, min_expected
                );
            });
        },
        py::arg("data"),
        py::arg("means"),
        py::arg("std_devs"),
        py::arg("min_pop") = 25,
        py::arg("seed") = 0,
        py::arg("min_expected") = 5.0,
        "Same as mi_normal_crossfit, but accepts a long-int matrix and casts internally."
    );

    m.def(
        "mi_negative_binomial_crossfit",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           int min_pop,
           std::uint64_t seed,
           double min_expected,
           bool use_empirical_marginals,
           double empirical_pseudocount)
           -> std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> {

            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd concentrations_d = as_vector_xd(concentrations, "concentrations");

            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(concentrations_d, "concentrations");

            if (data_i.cols() < 2) {
                throw py::value_error("data must contain at least two columns");
            }
            if (data_i.cols() != means_d.size() || data_i.cols() != concentrations_d.size()) {
                throw py::value_error("data columns must match length of means and concentrations");
            }

            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(concentrations_d, "concentrations");
            check_positive(means_d, "means");
            check_positive(concentrations_d, "concentrations");

            check_min_pop(min_pop);
            check_min_expected(min_expected);

            if (!(empirical_pseudocount >= 0.0) || !std::isfinite(empirical_pseudocount)) {
                throw py::value_error("empirical_pseudocount must be finite and >= 0");
            }

            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_nb_crossfit(
                    data_i,
                    means_d,
                    concentrations_d,
                    min_pop,
                    seed,
                    min_expected,
                    use_empirical_marginals,
                    empirical_pseudocount
                );
            });
        },
        py::arg("data"),
        py::arg("means"),
        py::arg("concentrations"),
        py::arg("min_pop") = 25,
        py::arg("seed") = 0,
        py::arg("min_expected") = 5.0,
        py::arg("use_empirical_marginals") = false,
        py::arg("empirical_pseudocount") = 0.5,
        "Cross-fit (2-fold) independence test with KD-tree partitioning.\n\n"
        "Returns (mi, chi2_cv, dof_cv), each (F,F) upper-triangular.\n"
        "By default the null uses NB marginals from means/concentrations.\n"
        "If use_empirical_marginals=True, fold-specific empirical marginals are used instead,\n"
        "with optional empirical_pseudocount smoothing.\n"
    );

    m.def(
        "mi_zero_inflated_negative_binomial_crossfit",
        [](const IntMatrixArray& data,
           const DoubleVectorArray& means,
           const DoubleVectorArray& concentrations,
           const DoubleVectorArray& alphas,
           int min_pop,
           std::uint64_t seed,
           double min_expected)
           -> std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> {

            Eigen::MatrixXi data_i = as_matrix_xi(data, "data");
            Eigen::VectorXd means_d = as_vector_xd(means, "means");
            Eigen::VectorXd concentrations_d = as_vector_xd(concentrations, "concentrations");
            Eigen::VectorXd alphas_d = as_vector_xd(alphas, "alphas");

            check_matrix_shape(data_i, "data");
            check_vector_size(means_d, "means");
            check_vector_size(concentrations_d, "concentrations");
            check_vector_size(alphas_d, "alphas");

            if (data_i.cols() < 2) {
                throw py::value_error("data must contain at least two columns");
            }
            if (data_i.cols() != means_d.size() ||
                data_i.cols() != concentrations_d.size() ||
                data_i.cols() != alphas_d.size()) {
                throw py::value_error(
                    "data columns must match length of means, concentrations, and alphas");
            }

            check_non_negative(data_i, "data");
            check_all_finite(means_d, "means");
            check_all_finite(concentrations_d, "concentrations");
            check_all_finite(alphas_d, "alphas");
            check_positive(means_d, "means");
            check_positive(concentrations_d, "concentrations");
            check_probabilities(alphas_d, "alphas");
            check_min_pop(min_pop);
            check_min_expected(min_expected);

            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_zinb_crossfit(
                    data_i, means_d, concentrations_d, alphas_d, min_pop, seed, min_expected
                );
            });
        },
        py::arg("data"),
        py::arg("means"),
        py::arg("concentrations"),
        py::arg("alphas"),
        py::arg("min_pop") = 25,
        py::arg("seed") = 0,
        py::arg("min_expected") = 5.0,
        "Cross-fit (2-fold) Pearson chi-square for ZINB copula-independence null.\n\n"
        "Returns (mi, chi2_cv, dof_cv), each (F,F) upper-triangular.\n"
        "chi2_cv and dof_cv are computed by building the KD-tree on fold A and\n"
        "evaluating chi-square on fold B (with expected-count bin merging), then swapping."
    );
}
