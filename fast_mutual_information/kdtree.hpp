#pragma once

#include <vector>
#include <cmath>
#include <numeric>

#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <functional>

#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>

// #include "absl/container/flat_hash_map.h"

#include "copula.hpp"
#include "point.hpp"
#include "utils.hpp"

template <typename T>
struct KDNode {

    // Use bounds struct here
    Bounds<T> bounds;

    std::vector<std::pair<Point<T>, int>> points;

    bool is_leaf;
    int split_dim; // 0 for x, 1 for y
    T split_val;

    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode() : is_leaf(false), split_dim(0), split_val(0) {
        bounds = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};
    }

    T get_width(void) const {
        return bounds.max_x - bounds.min_x;
    }

    T get_height(void) const {
        return bounds.max_y - bounds.min_y;
    }

    int total_counts(void) const;

    double get_bin_area(void) const {
        return this->get_width() * this->get_height();
    }

};

namespace std {
    template <>
    struct hash<Point<int>> {
        std::size_t operator()(const Point<int>& p) const {
            auto h1 = std::hash<int>()(p.x);
            auto h2 = std::hash<int>()(p.y);
            // Combine the two hash values.
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };
}

template <typename T>
int KDNode<T>::total_counts(void) const {
    // Here we assume that the points are unique, the second element of the pair is 1
    return this->points.size();
}

template <>
int KDNode<int>::total_counts(void) const {
    // Here we assume that the points are maybe not unique, and the second element of the pair could be greater than 1
    int count = 0;
    for (auto p : this->points) {
        count += p.second;
    }

    return count;
}



template <typename T>
class KDTree {
public:

    KDTree() {}

    KDTree(const std::vector<Point<T>>& points, Copula * copula, int max_points_per_leaf = 10);

    KDTree( std::vector<std::pair<Point<int>, int>> & unique_points, int nPoints, Bounds<int> bounds, Copula * copula, int max_points_per_leaf);

    double get_correction() const {
        int depth = get_tree_depth();

        int bins_xy = std::pow(2, depth);
        int bins_x = std::pow(2, depth / 2);
        int bins_y = std::pow(2, depth / 2);

        return (bins_xy - 1) / (2. * total_count);
    }

    // Function to compute mutual information
    double compute_mutual_information() const {
        double mi = 0.0;
        double area = 0.0;

        traverse_and_compute(root.get(), mi, area);

        return mi;
    }

    int calculate_depth(const KDNode<T>* node) const {
        if (!node) return 0;
        if (node->is_leaf) return 1;

        int left_depth = calculate_depth(node->left.get());
        int right_depth = calculate_depth(node->right.get());

        return 1 + std::max(left_depth, right_depth);
    }

    int get_tree_depth() const {
        return calculate_depth(root.get());
    }

private:
    std::unique_ptr<KDNode<T>> root;
    int max_points;
    int total_count;

    Copula * copula;

    double get_bin_area(const KDNode<T> & node) const;

    // For ints this can be optimised by sorting!

    // std::vector<std::pair<Point<int>, int>> count_duplicates_absl(const std::vector<Point<int>>& points) {

    //     absl::flat_hash_map<std::pair<int, int>, int, absl::Hash<std::pair<int, int>>> point_map;

    //     point_map.reserve(points.size() / 2);

    //     for (const auto& pt : points) {
    //         std::pair<int, int> key = {pt.x, pt.y};
    //         point_map[key]++;
    //     }

    //     std::vector<std::pair<Point<int>, int>> unique_points;
    //     unique_points.reserve(point_map.size());

    //     for (const auto& entry : point_map) {
    //         unique_points.emplace_back(std::make_pair(Point<T>{entry.first.first, entry.first.second}, entry.second));
    //     }

    //     return unique_points;
    // }

    std::vector<std::pair<Point<int>, int>> count_duplicates_unordered(const std::vector<Point<int>>& points) {
        std::unordered_map<Point<int>, int> counts;
        for (const auto& pt : points) {
            ++counts[pt];
        }

        std::vector<std::pair<Point<int>, int>> result;
        result.reserve(counts.size());
        for (const auto& entry : counts) {
            result.emplace_back(entry.first, entry.second);
        }

        // These will be sorted when splitting anyway

        // To match the lexicographical order from the sorting version.
        // std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        //     return a.first < b.first;
        // });

        return result;
    }

    std::vector<std::pair<Point<int>, int>> count_duplicates_sorted(const std::vector<Point<int>>& points) {
        if (points.empty()) return {};

        auto sorted_points = points;

        std::sort(sorted_points.begin(), sorted_points.end(), [](const Point<int>& a, const Point<int>& b) {
            return (a.x < b.x) || ((a.x == b.x) && (a.y < b.y));
        });

        // std::stable_sort(sorted_points.begin(), sorted_points.end(), [](const Point<int>& a, const Point<int>& b) {
        //     if (a.x != b.x)
        //         return a.x < b.x;
        //     return a.y < b.y;
        // });

        // boost::sort::spreadsort::integer_sort(sorted_points.begin(), sorted_points.end(),
        //                                      boost::sort::spreadsort::integer_traits<Point<int>>::base());


        std::vector<std::pair<Point<int>, int>> result;
        Point<int> current = sorted_points[0];
        int count = 1;
        for (std::size_t i = 1; i < sorted_points.size(); ++i) {
            if (sorted_points[i].x == current.x && sorted_points[i].y == current.y) {
                ++count;
            } else {
                result.emplace_back(current, count);
                current = sorted_points[i];
                count = 1;
            }
        }
        result.emplace_back(current, count);
        return result;
    }


    std::unique_ptr<KDNode<T>> build(std::vector<std::pair<Point<T>, int>>& points,
                                  int depth, Bounds<T> bounds) {

        auto node = std::make_unique<KDNode<T>>();
        node->bounds = bounds;

        bool degenerate_split = (bounds.max_x - bounds.min_x < 1E-8) || (bounds.max_y - bounds.min_y < 1E-8);

        // This is controlled by the number of points rather than the number of points including the duplicates as we don't want to end up with nowhere to split
        if (points.size() <= static_cast<size_t>(max_points) || degenerate_split) {
            node->is_leaf = true;
            node->points = points;
            return node;
        }

        int axis = depth % 2;
        node->split_dim = axis;

        if (axis == 0) {
            std::nth_element(points.begin(), points.begin() + points.size() / 2, points.end(),
                      [](const std::pair<Point<T>, int>& a, const std::pair<Point<T>, int>& b) -> bool {
                          return a.first.x < b.first.x;
                      });
        } else {
            std::nth_element(points.begin(), points.begin() + points.size() / 2, points.end(),
                      [](const std::pair<Point<T>, int>& a, const std::pair<Point<T>, int>& b) -> bool {
                          return a.first.y < b.first.y;
                      });
        }


        size_t median_idx = points.size() / 2;
        T median_val = (axis == 0) ? points[median_idx].first.x
                                   : points[median_idx].first.y;
        node->split_val = median_val;

        std::vector<std::pair<Point<T>, int>> left_points;
        std::vector<std::pair<Point<T>, int>> right_points;

        for (const auto& p : points) {
            int coord = (axis == 0) ? p.first.x : p.first.y;
            if (coord < median_val) {
                left_points.emplace_back(p);
            } else if (coord > median_val) {
                right_points.emplace_back(p);
            } else {
                if (left_points.size() <= right_points.size()) {
                    left_points.emplace_back(p);
                } else {
                    right_points.emplace_back(p);
                }
            }
        }

        // Handle potential empty subsets by enforcing the bounding box split

        Bounds<T> left_bounds;
        left_bounds.min_x = bounds.min_x;
        left_bounds.max_x = (axis == 0 ? median_val : bounds.max_x);
        left_bounds.min_y = bounds.min_y;
        left_bounds.max_y = (axis == 1 ? median_val : bounds.max_y);

        if (!left_points.empty()) {
            node->left = build(left_points, depth + 1, bounds);
        } else {
            auto leaf = std::make_unique<KDNode<T>>();
            leaf->is_leaf = true;
            leaf->bounds = left_bounds;
            node->left = std::move(leaf);
        }

        Bounds<T> right_bounds;
        right_bounds.min_x = (axis == 0 ? median_val : bounds.min_x);
        right_bounds.max_x = bounds.max_x;
        right_bounds.min_y = (axis == 1 ? median_val : bounds.min_y);
        right_bounds.max_y = bounds.max_y;

        if (!right_points.empty()) {
            node->right = build(right_points, depth + 1, bounds);
        } else {
            auto leaf = std::make_unique<KDNode<T>>();
            leaf->is_leaf = true;
            leaf->bounds = right_bounds;
            node->right = std::move(leaf);
        }

        return node;
    }

    int compute_total_count(const KDNode<T>* node) const {
        if (!node) return 0;
        if (node->is_leaf) {
            return node->total_counts();
        }
        return compute_total_count(node->left.get()) + compute_total_count(node->right.get());
    }

    // Can I make the underlying storage here an eigen vector, and then just push it through the NB calculation? Or maybe even populate it with points and the corresponding NB beforehand? -> Take the two Eigen vectors, calculate the NB, and then pop the points with (x, y, nb_x, nb_y)

    double calculate_p_xy(int count, double bin_area) const {
        return static_cast<double>(count) / (static_cast<double>(total_count) * bin_area);
    }

    void traverse_and_compute(const KDNode<T>* node, double& mi, double &area) const {
        if (!node) return;

        if (node->is_leaf) {

            int bin_count = node->total_counts();

            double bin_area = this->get_bin_area(*node);

            double p_xy = calculate_p_xy(bin_count, bin_area);

            if (p_xy > 0) {
                mi += p_xy * std::log(p_xy) * bin_area;
            }

            area += bin_area;

            return;
        }

        traverse_and_compute(node->left.get(), mi, area);
        traverse_and_compute(node->right.get(), mi, area);
    }
};

template <typename T>
KDTree<T>::KDTree(const std::vector<Point<T>>& points, Copula * copula, int max_points_per_leaf)
    : max_points(max_points_per_leaf), copula(copula) {

    total_count = points.size();

    std::vector<std::pair<Point<T>, int>> unique_points;

    for (int i = 0; i < points.size(); i++) {
        unique_points.push_back(std::make_pair(points[i], 1));
    }

    // These are the boundaries of the unit square for assumed U[0, 1] if passed anything other than ints

    Bounds<T> bounds = {0.0, 1.0, 0.0, 1.0};

    root = build(unique_points, 0, bounds);
}

template <>
KDTree<int>::KDTree(const std::vector<Point<int>>& points, Copula * copula, int max_points_per_leaf)
    : max_points(max_points_per_leaf), copula(copula) {

    total_count = points.size();

    std::vector<std::pair<Point<int>, int>> unique_points = count_duplicates_unordered(points);

    // These are the boundaries of the input data that then get mapped to [0, 1, 0, 1] when transformed via the CDF

    Bounds<int> bounds = get_bounds(points);

    root = build(unique_points, 0, bounds);
}

// For pre-calculated duplicates on RLE vectors - would be nice to make this const, but then it has to be sorted
template <>
KDTree<int>::KDTree( std::vector<std::pair<Point<int>, int>> & unique_points, int nPoints, Bounds<int> bounds, Copula * copula, int max_points_per_leaf)
    : max_points(max_points_per_leaf), copula(copula) {

    root = build(unique_points, 0, bounds);
}

template <typename T>
double KDTree<T>::get_bin_area(const KDNode<T> & node) const {
    return node.get_bin_area();
}

template <>
double KDTree<int>::get_bin_area(const KDNode<int> & node) const {

    // Transform to the uniform distribution via the CDF, to get
    // the area in U[0, 1] space

    double x_min = this->copula->cdf_x(node.bounds.min_x);
    double x_max = this->copula->cdf_x(node.bounds.max_x);

    double y_min = this->copula->cdf_y(node.bounds.min_y);
    double y_max = this->copula->cdf_y(node.bounds.max_y);

    double width = x_max - x_min;
    double height = y_max - y_min;

    return width * height;

}
