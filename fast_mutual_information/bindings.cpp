#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fast_mi.cpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_mutual_information, m) {
    m.doc() = "Python bindings for fast mutual information computation.";

    // m.def(
    //     "negative_binomial",
    //     [](int k, int r, double p) -> double { return nb_base<int>(k, r, p); },
    //     py::arg("k"), py::arg("r"), py::arg("p"),
    //     "Compute the Negative Binomial PMF.\n\n"
    //     "Parameters:\n"
    //     "    k (int): Number of failures.\n"
    //     "    r (int): Number of successes.\n"
    //     "    p (float): Probability of success on an individual trial.\n\n"
    //     "Returns:\n"
    //     "    float: The PMF value.");

}
