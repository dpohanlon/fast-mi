#pragma once

#include <functional>
#include <string>

template<typename T>
struct DistributionFunctions {
    // Sometimes also a PMF depending on the type
    using pdf_f = std::function<double(T)>;
    using cdf_f = std::function<double(T)>;
    using icdf_f = std::function<T(double)>;
};

template<typename T>
class Copula {
public:

    using pdf_f  = typename DistributionFunctions<T>::pdf_f;
    using cdf_f  = typename DistributionFunctions<T>::cdf_f;
    using icdf_f = typename DistributionFunctions<T>::icdf_f;

    Copula() {}

    Copula(pdf_f p_x, pdf_f p_y, cdf_f cdf_x, cdf_f cdf_y, icdf_f icdf_x, icdf_f icdf_y)
        : p_x(p_x), p_y(p_y), cdf_x(cdf_x), cdf_y(cdf_y), icdf_x(icdf_x), icdf_y(icdf_y) {}

    std::string name = "UNINITIALIZED";

    // PDFs
    pdf_f p_x;
    pdf_f p_y;

    // CDFs
    cdf_f cdf_x;
    cdf_f cdf_y;

    // Inverse CDFs
    icdf_f icdf_x;
    icdf_f icdf_y;
};
