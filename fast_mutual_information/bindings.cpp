#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mutual_information.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_mutual_information, m) {
    m.doc() = "Python bindings for fast mutual information computation.";

    m.def(
        "mi_normal",
        [](double mean1, double std_dev1, double mean2, double std_dev2,
           Eigen::MatrixXd& data, int min_pop) -> double {
            return mutual_information_normal(mean1, std_dev1, mean2, std_dev2,
                                             data, min_pop);
        },
        py::arg("mean1"), py::arg("mean2"), py::arg("std_dev1"),
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
        [](Eigen::MatrixXd& samples, Eigen::VectorXd means,
           Eigen::VectorXd variances, int min_pop) -> Eigen::MatrixXd {
            return mutual_information_normal(samples, means, variances,
                                             min_pop);
        },
        py::arg("data"), py::arg("means"), py::arg("variances"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with normally distributed "
        "marginals.\n\n"
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
        "mi_normal_q",
        [](Eigen::MatrixXi& data, Eigen::VectorXd& means,
           Eigen::VectorXd& variances, int min_pop) -> Eigen::MatrixXd {
            return mutual_information_normal(data, means, variances, min_pop);
        },
        py::arg("data"), py::arg("means"), py::arg("variances"),
        py::arg("min_pop") = 25,
        "Fast mutual information computation with normally distributed "
        "marginals.\n\n"
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
    // Also only the run length encoding version, so strictly only integer data

    m.def(
        "mi_negative_binomial",
        [](Eigen::MatrixXi& data, Eigen::VectorXd& means,
           Eigen::VectorXd& concentrations, int min_pop) -> Eigen::MatrixXd {
            return mutual_information_nb_rle(data, means, concentrations,
                                             min_pop);
        },
        py::arg("data"), py::arg("means"), py::arg("variances"),
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
}
