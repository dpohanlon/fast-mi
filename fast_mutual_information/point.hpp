#pragma once

#include <Eigen/Dense>

template <typename T>
struct Point {
    T x;
    T y;

    bool operator<(const Point& other) const {
        return (x < other.x) || ((x == other.x) && (y < other.y));
    }

    bool operator==(const Point& other) const {
        return (x == other.x) && (y == other.y);
    }
};

// In your utils.hpp (or wherever you like):

template<typename Scalar>
struct PointIterator {
    using value_type = Point<Scalar>;
    using pointer    = void;
    using reference  = value_type;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    const Scalar* px;
    const Scalar* py;

    PointIterator(const Scalar* x, const Scalar* y)
      : px(x), py(y) {}

    // advance
    PointIterator& operator++() {
        ++px; ++py;
        return *this;
    }
    // compare
    bool operator!=(PointIterator o) const {
        return px != o.px;
    }
    // dereference
    value_type operator*() const {
        return Point<Scalar>{*px, *py};
    }
};

template<typename ColVec>
struct PointView {
    ColVec xs, ys;  // own the Eigen proxy

    PointView(ColVec x, ColVec y)
      : xs(x), ys(y)
    {
        if (xs.size() != ys.size())
            throw std::invalid_argument("PointView: mismatched lengths");
    }

    using Scalar   = typename ColVec::Scalar;
    using iterator = PointIterator<Scalar>;

    iterator begin() const { return { xs.data(), ys.data() }; }
    iterator end()   const { return { xs.data() + xs.size(),
                                       ys.data() + ys.size() }; }
};
