#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mutual_information.hpp"

// Looong looong matrix
typedef Eigen::Matrix<long int, Eigen::Dynamic, Eigen::Dynamic> MatrixXl;

namespace py = pybind11;

PYBIND11_MODULE(fast_mutual_information, m) {
    m.doc() = "Python bindings for fast mutual information computation.";

    m.def(
        "mi_ml",
        [](MatrixXl& data) -> Eigen::MatrixXd {
            Eigen::MatrixXi data_i = data.cast<int>().eval();
            return mutual_information_ml(data_i);
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
            return mutual_information_binarised(samples);
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
            return mutual_information_normal(mean1, std_dev1, mean2, std_dev2,
                                             data, min_pop);
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

    m.def("mi_normal",
          [](const Eigen::MatrixXd& data, const Eigen::VectorXd& means,
             const Eigen::VectorXd& std_devs, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
              return mutual_information_normal(data, means, std_devs, min_pop);
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
          "    std_devs (np.array): Standard deviations of each normal distribution "
          "(Nfeatures, 1).\n"
          "    min_pop (int): Mininmum bin population.\n\n"
          "Returns:\n"
          "    np.array: Array of mutual information values.");

    m.def(
        "mi_normal",
        [](Eigen::MatrixXi& data, Eigen::VectorXd& means,
           Eigen::VectorXd& variances, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            return mutual_information_normal(data, means, variances, min_pop);
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
        [](MatrixXl& data, Eigen::VectorXd& means,
           Eigen::VectorXd& variances, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            Eigen::MatrixXi data_i = data.cast<int>().eval();
            return mutual_information_normal(data_i, means, variances, min_pop);
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
        [](Eigen::MatrixXi& data, Eigen::VectorXd& means,
           Eigen::VectorXd& concentrations, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{
            return mutual_information_nb(data, means, concentrations,
                                             min_pop);
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
        "mi_negative_binomial_zi",
        [](Eigen::MatrixXi& data, Eigen::VectorXd& means,
           Eigen::VectorXd& concentrations, Eigen::VectorXd& alphas, int min_pop) -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd> {
            return mutual_information_zinb(data, means, concentrations, alphas,
                                             min_pop);
        },
        py::arg("data"), py::arg("means"), py::arg("concentrations"), py::arg("alphas"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with negative binomial "
        "marginals.\n\n"
        "Parameters:\n"
        "    data (np.array): Integer data array of shape (Nsamples, "
        "(Nfeatures, 1).\n"
        "    means (np.array): Means of each negative binomial distribution "
        "(Nfeatures, 1).\n"
        "    concentrations (np.array): concentrations of each negative "
        "binomial distribution (Nfeatures, 1).\n"
        "    alphas (np.array): Zero inflation probabilities (alpha -> 0, pure NB) "
        "(Nfeatures, 1).\n"
        "    min_pop (int): Mininmum bin population.\n\n"
        "Returns:\n"
        "    np.array: Array of mutual information values.");

}
