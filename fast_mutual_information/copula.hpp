#pragma once

#include<functional>

typedef std::function<double(int)> pmf_f;
typedef std::function<double(int)> cdf_f;
typedef std::function<int(double)> icdf_f;

class Copula {

public:
    Copula(pmf_f &p_x, pmf_f p_y, cdf_f &cdf_x, cdf_f &cdf_y, icdf_f &icdf_x, icdf_f &icdf_y) : p_x(p_x), p_y(p_y), cdf_x(cdf_x), cdf_y(cdf_y), icdf_x(icdf_x), icdf_y(icdf_y) {}

    // PDFs
    pmf_f p_x;
    pmf_f p_y;

    // CDFs
    cdf_f cdf_x;
    cdf_f cdf_y;

    // Inverse CDFs
    icdf_f icdf_x;
    icdf_f icdf_y;

};
