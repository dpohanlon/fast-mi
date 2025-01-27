#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fast_mi.cpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_mutual_information, m) {
    m.doc() = "Python bindings for fast mutual information computation.";

    m.def(
        "mi_normal",
        [](double mean1, double std_dev1, double mean2, double std_dev2, Eigen::MatrixXd & data) -> double { return mutual_information_normal(mean1, std_dev1, mean2, std_dev2, data); },
        py::arg("mean1"), py::arg("mean2"), py::arg("std_dev1"), py::arg("std_dev2"), py::arg("data"),
        "Fast mutual information computation with normally distributed marginals.\n\n"
        "Parameters:\n"
        "    mean1 (float): Mean of the first axis normal distribution.\n"
        "    std_dev1 (float): Standard-deviation of the first axis normal distribution.\n"
        "    mean2 (float): Mean of the second axis normal distribution\n"
        "    std_dev2 (float): Standard-deviation of the second axis normal distribution.\n"
        "    data (np.array): Data array of shape (Nsamples, 2).\n\n"
        "Returns:\n"
        "    float: The mutual information.");

}
