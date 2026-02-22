#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
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

}  // namespace

PYBIND11_MODULE(fast_mutual_information, m) {
    m.doc() = "Python bindings for fast mutual information computation.";

    m.def(
        "mi_ml",
        [](MatrixXl& data) -> Eigen::MatrixXd {
            Eigen::MatrixXi data_i = data.cast<int>().eval();
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
        [](Eigen::MatrixXi& samples) -> Eigen::MatrixXd {
            check_matrix_shape(samples, "samples");
            if (samples.cols() < 2) {
                throw py::value_error(
                    "samples must contain at least two columns");
            }
            return safe_execute(
                [&] { return mutual_information_binarised(samples); });
        },
        py::arg("samples"),
        "Fast binarised mutual information computation.\n\n"
        "    data (np.array): Data array of shape (Nsamples, Nfeatures).\n"
        "Returns:\n"
        "    float: The mutual information.");

    m.def(
        "mi_normal",
        [](double mean1, double std_dev1, double mean2, double std_dev2,
           Eigen::MatrixXd& data, int min_pop) -> std::pair<double, double> {
            check_matrix_shape(data, "data");
            check_all_finite(data, "data");
            if (data.cols() != 2) {
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
                                                 std_dev2, data, min_pop);
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
        [](const Eigen::MatrixXd& data, const Eigen::VectorXd& means,
           const Eigen::VectorXd& std_devs,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            check_matrix_shape(data, "data");
            check_vector_size(means, "means");
            check_vector_size(std_devs, "std_devs");
            check_all_finite(data, "data");
            check_all_finite(means, "means");
            check_all_finite(std_devs, "std_devs");
            if (data.cols() != means.size() || data.cols() != std_devs.size()) {
                throw py::value_error(
                    "data columns must match length of means and std_devs");
            }
            check_positive(std_devs, "std_devs");
            check_min_pop(min_pop);
            return safe_execute([&] {
                return mutual_information_normal(data, means, std_devs,
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
        [](Eigen::MatrixXi& data, Eigen::VectorXd& means,
           Eigen::VectorXd& variances,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            check_matrix_shape(data, "data");
            check_vector_size(means, "means");
            check_vector_size(variances, "variances");
            if (data.cols() != means.size() ||
                data.cols() != variances.size()) {
                throw py::value_error(
                    "data columns must match length of means and variances");
            }
            check_non_negative(data, "data");
            check_all_finite(means, "means");
            check_all_finite(variances, "variances");
            check_positive(variances, "variances");
            check_min_pop(min_pop);
            return safe_execute([&] {
                return mutual_information_normal(data, means, variances,
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
        [](MatrixXl& data, Eigen::VectorXd& means, Eigen::VectorXd& variances,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXi data_i = data.cast<int>().eval();
            check_matrix_shape(data_i, "data");
            check_vector_size(means, "means");
            check_vector_size(variances, "variances");
            if (data_i.cols() != means.size() ||
                data_i.cols() != variances.size()) {
                throw py::value_error(
                    "data columns must match length of means and variances");
            }
            check_non_negative(data_i, "data");
            check_all_finite(means, "means");
            check_all_finite(variances, "variances");
            check_positive(variances, "variances");
            check_min_pop(min_pop);
            return safe_execute([&] {
                return mutual_information_normal(data_i, means, variances,
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
        [](py::EigenDRef<const Eigen::MatrixXi> data, py::EigenDRef<const Eigen::VectorXd> means,
           py::EigenDRef<const Eigen::VectorXd> concentrations,
           int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            check_matrix_shape(data, "data");
            check_vector_size(means, "means");
            check_vector_size(concentrations, "concentrations");
            if (data.cols() != means.size() ||
                data.cols() != concentrations.size()) {
                throw py::value_error(
                    "data columns must match length of means and "
                    "concentrations");
            }
            check_non_negative(data, "data");
            check_all_finite(means, "means");
            check_all_finite(concentrations, "concentrations");
            check_positive(means, "means");
            check_positive(concentrations, "concentrations");
            check_min_pop(min_pop);

            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_nb(data, means, concentrations,
                                             min_pop);
            });
        },
        py::arg("data").noconvert(), py::arg("means").noconvert(), py::arg("concentrations").noconvert(),
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
        [](Eigen::MatrixXi& data, Eigen::VectorXd& means,
           Eigen::VectorXd& concentrations, Eigen::VectorXd& alphas,
           const std::string& csv_filename,
           int min_pop) -> int {
            check_matrix_shape(data, "data");
            check_vector_size(means, "means");
            check_vector_size(concentrations, "concentrations");
            check_vector_size(alphas, "alphas");
            if (data.cols() != means.size() ||
                data.cols() != concentrations.size() ||
                data.cols() != alphas.size()) {
                throw py::value_error(
                    "data columns must match length of means, concentrations, and alphas");
            }
            check_non_negative(data, "data");
            check_all_finite(means, "means");
            check_all_finite(concentrations, "concentrations");
            check_all_finite(alphas, "alphas");
            check_positive(means, "means");
            check_positive(concentrations, "concentrations");
            check_min_pop(min_pop);

            safe_execute([&] {
                return mutual_information_zinb_dump_first_tree(
                    data, means, concentrations, alphas, csv_filename, min_pop);
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
        [](py::EigenDRef<const Eigen::MatrixXi> data, py::EigenDRef<const Eigen::VectorXd> means, py::EigenDRef<const Eigen::VectorXd> concentrations, py::EigenDRef<const Eigen::VectorXd> alphas,
        int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            check_matrix_shape(data, "data");
            check_vector_size(means, "means");
            check_vector_size(concentrations, "concentrations");
            check_vector_size(alphas, "alphas");
            if (data.cols() != means.size() ||
                data.cols() != concentrations.size() ||
                data.cols() != alphas.size()) {
                throw py::value_error(
                    "data columns must match length of means, concentrations "
                    "and alphas");
            }
            check_non_negative(data, "data");
            check_all_finite(means, "means");
            check_all_finite(concentrations, "concentrations");
            check_all_finite(alphas, "alphas");
            check_positive(means, "means");
            check_positive(concentrations, "concentrations");
            check_probabilities(alphas, "alphas");
            check_min_pop(min_pop);
            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_zinb(data, means, concentrations,
                                               alphas, min_pop);
            });
        },
        py::arg("data").noconvert(), py::arg("means").noconvert(), py::arg("concentrations").noconvert(),
        py::arg("alphas").noconvert(), py::arg("min_pop") = 25,
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
        [](py::EigenDRef<const Eigen::MatrixXi> data, py::EigenDRef<const Eigen::VectorXd> means, py::EigenDRef<const Eigen::VectorXd> concentrations, py::EigenDRef<const Eigen::VectorXd> exposure,
        int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {

            check_matrix_shape(data, "data");
            check_vector_size(means, "means");
            check_vector_size(concentrations, "concentrations");
            check_vector_size(exposure, "exposure");
            if (data.cols() != means.size() ||
                data.cols() != concentrations.size()) {
                throw py::value_error(
                    "data columns must match length of means, concentrations");
            }
            if (data.rows() != exposure.size()) {
                throw py::value_error(
                    "data rows must match length of exposure");
            }
            check_non_negative(data, "data");
            check_all_finite(means, "means");
            check_all_finite(concentrations, "concentrations");
            check_all_finite(exposure, "exposure");
            check_positive(means, "means");
            check_positive(concentrations, "concentrations");
            check_positive(exposure, "exposure");
            check_min_pop(min_pop);

            return safe_execute([&] {
                py::gil_scoped_release nogil;
                return mutual_information_nb(data, means, concentrations,
                                               exposure, min_pop);
            });

        },
        py::arg("data").noconvert(),
        py::arg("means").noconvert(),
        py::arg("concentrations").noconvert(),
        py::arg("exposure").noconvert(),
        py::arg("min_pop") = 25,
        "Fast MI with NB marginals **and per-sample exposure offsets**.\n"
        "Interpret 'means_mu0' as baseline means on unit exposure; the per-sample\n"
        "mean is mu_j = mu0 * exposure[j]. The marginal CDF used by the kd-tree is\n"
        "the mixture F_mix(k) = mean_j F_NB(k; mu0*exposure[j], r)."
    );
    // Sparse matrix versions

    m.def(
        "mi_normal_sparse",
        [](Eigen::SparseMatrix<int>& data, Eigen::VectorXd& means,
           Eigen::VectorXd& std_devs, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            return mutual_information_normal_sparse(data, means, std_devs, min_pop);
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
        [](Eigen::SparseMatrix<int>& data, Eigen::VectorXd& means,
           Eigen::VectorXd& concentrations, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            return mutual_information_nb_sparse(data, means, concentrations, min_pop);
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
}
