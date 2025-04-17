#pragma once

#include <algorithm>
#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <fstream>

#include "copula.hpp"
#include "point.hpp"
#include "utils.hpp"

// #define DEBUG

template <typename T>
struct KDNode {

    Bounds<T> bounds;

    std::vector<std::pair<Point<T>, int>> points;

    bool is_leaf;
    int split_dim;  // 0 for x, 1 for y
    T split_val;

    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode() : is_leaf(false), split_dim(0), split_val(0) {
        bounds = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};
    }

    T get_width(void) const { return bounds.max_x - bounds.min_x; }

    T get_height(void) const { return bounds.max_y - bounds.min_y; }

    int total_counts(void) const;

    double get_bin_area(void) const {
        return this->get_width() * this->get_height();
    }
};

template <>
int KDNode<int>::get_width(void) const {
    return std::max(1, bounds.max_x - bounds.min_x);
}

template <>
int KDNode<int>::get_height(void) const {
    return std::max(1, bounds.max_y - bounds.min_y);
}

template <typename T>
int KDNode<T>::total_counts(void) const {
    // Here we assume that the points are unique, the second element of the pair
    // is 1
    return this->points.size();
}

template <>
int KDNode<int>::total_counts(void) const {
    // Here we assume that the points are maybe not unique, and the second
    // element of the pair could be greater than 1
    int count = 0;
    for (auto p : this->points) {
        count += p.second;
    }

    return count;
}

namespace std {
template <>
struct hash<Point<int>> {
    std::size_t operator()(const Point<int>& p) const {
        auto h1 = std::hash<int>()(p.x);
        auto h2 = std::hash<int>()(p.y);

        // More magic
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};
}  // namespace std

template <typename T>
class KDTree {
   public:
    KDTree() {}

    KDTree(const std::vector<Point<T>>& points, Copula<T>* copula,
           int min_points_per_leaf = 10, bool zi = false);

    KDTree(std::vector<std::pair<Point<int>, int>>& unique_points, int nPoints,
           Bounds<int> bounds, Copula<int>* copula, int min_points_per_leaf, bool zi = false);

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

    int get_tree_depth() const { return calculate_depth(root.get()); }

    void dumpSplittingValuesToCSV(const std::string &filename) const;

   private:
    std::unique_ptr<KDNode<T>> root;
    int min_points;
    int total_count;
    bool zi;

    Copula<T>* copula;

    double get_bin_area(const KDNode<T>& node) const;

    void dumpSplittingValuesHelper(const KDNode<T>* node, std::ofstream &out, int depth) const;

    // For ints this can be optimised by sorting!

    std::vector<std::pair<Point<int>, int>> count_duplicates_unordered(
        const std::vector<Point<int>>& points) {
        std::unordered_map<Point<int>, int> counts;
        for (const auto& pt : points) {
            ++counts[pt];
        }

        std::vector<std::pair<Point<int>, int>> result;
        result.reserve(counts.size());
        for (const auto& entry : counts) {
            result.emplace_back(entry.first, entry.second);
        }

        return result;
    }

    std::vector<std::pair<Point<int>, int>> count_duplicates_sorted(
        const std::vector<Point<int>>& points) {
        if (points.empty()) return {};

        auto sorted_points = points;

        std::sort(sorted_points.begin(), sorted_points.end(),
                  [](const Point<int>& a, const Point<int>& b) {
                      return (a.x < b.x) || ((a.x == b.x) && (a.y < b.y));
                  });

        std::vector<std::pair<Point<int>, int>> result;
        Point<int> current = sorted_points[0];
        int count = 1;
        for (std::size_t i = 1; i < sorted_points.size(); ++i) {
            if (sorted_points[i].x == current.x &&
                sorted_points[i].y == current.y) {
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

    int aggregate_count(const std::vector<std::pair<Point<T>, int>>& pts) {
        int sum = 0;
        for (const auto &p : pts) {
            sum += p.second;
        }
        return sum;
    };

    std::unique_ptr<KDNode<T>> build(
            std::vector<std::pair<Point<T>, int>>& points, int depth,
            Bounds<T> bounds, bool zi = false) {

        auto node = std::make_unique<KDNode<T>>();
        node->bounds = bounds;

        bool degenerate_split = (bounds.max_x - bounds.min_x < 1E-8) ||
                                (bounds.max_y - bounds.min_y < 1E-8);

        // This is controlled by the number of points rather than the number of
        // points including the duplicates as we don't want to end up with
        // nowhere to split
        if (points.size() <= static_cast<size_t>(min_points) ||
            degenerate_split) {
        // int total_agg = aggregate_count(points);
        // if (total_agg <= min_points || degenerate_split) {
            node->is_leaf = true;
            node->points = points;
            return node;
        }

        int axis = depth % 2;
        node->split_dim = axis;

        T median_val;
        if (false && zi && ((axis == 0 && depth == 0) || (axis == 1 && depth == 1))) {
            // Force a split between 0 and 1 (i.e., at 0.5) to separate zero from nonzero values.
            median_val = static_cast<T>(1);
        } else {
            if (axis == 0) {
                std::nth_element(points.begin(), points.begin() + points.size() / 2,
                                   points.end(),
                                   [](const std::pair<Point<T>, int>& a,
                                      const std::pair<Point<T>, int>& b) -> bool {
                                       return a.first.x < b.first.x;
                                   });
            } else {
                std::nth_element(points.begin(), points.begin() + points.size() / 2,
                                   points.end(),
                                   [](const std::pair<Point<T>, int>& a,
                                      const std::pair<Point<T>, int>& b) -> bool {
                                       return a.first.y < b.first.y;
                                   });
            }
            size_t median_idx = points.size() / 2;
            median_val = (axis == 0) ? points[median_idx].first.x : points[median_idx].first.y;
        }
        node->split_val = median_val;

        std::vector<std::pair<Point<T>, int>> left_points;
        std::vector<std::pair<Point<T>, int>> right_points;

        for (const auto& p : points) {
            T coord = (axis == 0 ? p.first.x : p.first.y);
            if (coord < median_val)
                left_points.emplace_back(p);
            else
                right_points.emplace_back(p);
        }

        if (left_points.empty() || right_points.empty()) {
            node->is_leaf = true;
            node->points  = points;      // keep original points
            return node;                 // stop splitting
        }

        // int left_agg = aggregate_count(left_points);
        // int right_agg = aggregate_count(right_points);
        // if (left_agg == 0 || right_agg == 0) {
        //     // The candidate split would leave one side empty, so do not split.
        //     node->is_leaf = true;
        //     node->points = points;
        //     return node;
        // }

        Bounds<T> left_bounds = bounds;
        Bounds<T> right_bounds = bounds;

        if (axis == 0) {
            if (std::is_integral<T>::value) {
                // half‐open integer split: left ≤ median_val–1, right ≥ median_val
                left_bounds .max_x = median_val - 1;
                right_bounds.min_x = median_val;
            } else {
                // floating‐point split remains closed on left, open on right
                left_bounds .max_x = median_val;
                right_bounds.min_x = median_val;
            }
        } else {
            if (std::is_integral<T>::value) {
                left_bounds .max_y = median_val - 1;
                right_bounds.min_y = median_val;
            } else {
                left_bounds .max_y = median_val;
                right_bounds.min_y = median_val;
            }
        }

        if (!left_points.empty()) {
            node->left = build(left_points, depth + 1, left_bounds, zi);
        }

        if (!right_points.empty()) {
            node->right = build(right_points, depth + 1, right_bounds, zi);
        }

        return node;
    }

    int compute_total_count(const KDNode<T>* node) const {
        if (!node) return 0;
        if (node->is_leaf) {
            return node->total_counts();
        }
        return compute_total_count(node->left.get()) +
               compute_total_count(node->right.get());
    }

    // Can I make the underlying storage here an eigen vector, and then just
    // push it through the NB calculation? Or maybe even populate it with points
    // and the corresponding NB beforehand? -> Take the two Eigen vectors,
    // calculate the NB, and then pop the points with (x, y, nb_x, nb_y)

    double calculate_p_xy(int count, double bin_area) const {
        return static_cast<double>(count) /
               (static_cast<double>(total_count) * bin_area);
    }

    void traverse_and_compute(const KDNode<T>* node, double& mi,
                              double& area) const {
        if (!node) return;

        if (node->is_leaf) {
            int bin_count = node->total_counts();
            if (bin_count == 0) return;

            double bin_area = this->get_bin_area(*node);

            const double epsilon = 1e-12;
            // if (bin_area < epsilon) {
            //     bin_area = epsilon;
            // }

            // double p_xy = calculate_p_xy(bin_count, bin_area);

            // if (p_xy > 0) {
            //     mi += p_xy * std::log(p_xy) * bin_area;
            // }

            // Compute log-density with a regularized bin area to avoid log(0)
            double log_p_xy = std::log(bin_count) - std::log(total_count) - std::log(bin_area + epsilon);

            // Accumulate mutual information contribution from this leaf
            mi += (static_cast<double>(bin_count) / total_count) * log_p_xy;

            area += bin_area;

            return;
        }

        traverse_and_compute(node->left.get(), mi, area);
        traverse_and_compute(node->right.get(), mi, area);
    }
};

template <typename T>
KDTree<T>::KDTree(const std::vector<Point<T>>& points, Copula<T>* copula,
                  int min_points_per_leaf, bool zi)
    : min_points(min_points_per_leaf), copula(copula), zi(zi) {
    total_count = points.size();

    std::vector<std::pair<Point<T>, int>> unique_points;

    for (int i = 0; i < points.size(); i++) {
        unique_points.push_back(std::make_pair(points[i], 1));
    }

    // These are the boundaries of the unit square for assumed U[0, 1] if passed
    // anything other than ints

    // Check that this is true for other normal mi where the uniform transform isn't done?

    Bounds<T> bounds = {0.0, 1.0, 0.0, 1.0};

    root = build(unique_points, 0, bounds, zi);

    #ifdef DEBUG
    this->dumpSplittingValuesToCSV("debug_splits.csv");
    #endif
}

template <>
KDTree<int>::KDTree(const std::vector<Point<int>>& points, Copula<int>* copula,
                    int min_points_per_leaf, bool zi)
    : min_points(min_points_per_leaf), copula(copula), zi(zi) {
    total_count = points.size();

    std::vector<std::pair<Point<int>, int>> unique_points =
        count_duplicates_unordered(points);

    // These are the boundaries of the input data that then get mapped to [0, 1,
    // 0, 1] when transformed via the CDF

    Bounds<int> bounds = get_bounds(points);

    root = build(unique_points, 0, bounds, zi);

    #ifdef DEBUG
    this->dumpSplittingValuesToCSV("debug_splits.csv");
    #endif
}

// For pre-calculated duplicates on RLE vectors - would be nice to make this
// const, but then it has to be sorted
template <>
KDTree<int>::KDTree(std::vector<std::pair<Point<int>, int>>& unique_points,
                    int nPoints, Bounds<int> bounds, Copula<int>* copula,
                    int min_points_per_leaf, bool zi)
    : min_points(min_points_per_leaf), copula(copula), zi(zi) {
    root = build(unique_points, 0, bounds, zi);

    #ifdef DEBUG
    this->dumpSplittingValuesToCSV("debug_splits.csv");
    #endif
}

template <typename T>
double KDTree<T>::get_bin_area(const KDNode<T>& node) const {
    return node.get_bin_area();
}


template <>
double KDTree<int>::get_bin_area(const KDNode<int>& node) const {
    // lower CDF edge = F(k-1), but clamp at zero
    int lo_x = node.bounds.min_x - 1;
    int lo_y = node.bounds.min_y - 1;
    double x_min = (lo_x >= 0 ? copula->cdf_x(lo_x) : 0.0);
    double y_min = (lo_y >= 0 ? copula->cdf_y(lo_y) : 0.0);

    // upper edge always = F(k)
    double x_max = copula->cdf_x(node.bounds.max_x);
    double y_max = copula->cdf_y(node.bounds.max_y);

    double width  = x_max - x_min;
    double height = y_max - y_min;
    return width * height;
}

template <typename T>
void KDTree<T>::dumpSplittingValuesToCSV(const std::string &filename) const {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening file for dump: " << filename << std::endl;
        return;
    }

    out << "depth,split_dim,split_val,is_leaf,min_x,max_x,min_y,max_y\n";
    dumpSplittingValuesHelper(root.get(), out, 0);
    out.close();
}

template <typename T>
void KDTree<T>::dumpSplittingValuesHelper(const KDNode<T>* node, std::ofstream &out, int depth) const {
    if (!node) return;

    // Output current node info:
    out << depth << ",";

    // For non-leaf nodes, write the splitting info; for leaf nodes, indicate not applicable.
    if (!node->is_leaf) {
        out << node->split_dim << "," << node->split_val;
    } else {
        out << "-1,NA"; // using -1 for split_dim and "NA" for split_val if leaf node.
    }
    out << "," << (node->is_leaf ? "true" : "false") << ",";
    // Output the bounding box values.
    out << node->bounds.min_x << "," << node->bounds.max_x << ",";
    out << node->bounds.min_y << "," << node->bounds.max_y << "\n";

    // Recursively dump left and right subtrees.
    dumpSplittingValuesHelper(node->left.get(), out, depth + 1);
    dumpSplittingValuesHelper(node->right.get(), out, depth + 1);
}
