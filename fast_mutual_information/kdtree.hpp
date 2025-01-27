#pragma once

#include <vector>
#include <cmath>
#include <numeric>

#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <functional>

#include "absl/container/flat_hash_map.h"

#include "copula.hpp"

template <typename T>
struct Point {
    T x;
    T y;
};

template <typename T>
struct KDNode {

    T min_x, max_x;
    T min_y, max_y;

    std::vector<std::pair<Point<T>, int>> points;

    bool is_leaf;
    int split_dim; // 0 for x, 1 for y
    T split_val;

    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode() : is_leaf(false), split_dim(0), split_val(0),
               min_x(1E8), max_x(-1E8),
               min_y(1E8), max_y(-1E8) {}

    T get_width(void) const {
        return max_x - min_x;
    }

    T get_height(void) const {
        return max_y - min_y;
    }

    int total_counts(void) const;

    double get_bin_area(void) const {
        return this->get_width() * this->get_height();
    }

};

template <typename T>
int KDNode<T>::total_counts(void) const {
    return this->points.size();
}

template <>
int KDNode<int>::total_counts(void) const {
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

    KDTree(const std::vector<Point<T>>& points, Copula * copula, int max_points_per_leaf = 10)
        : max_points(max_points_per_leaf), copula(copula) {

        long long sum_points = 0;
        for (auto &p : points) sum_points += 1; // or if using duplicates, sum up p.second
        total_count = sum_points;

        std::vector<std::pair<Point<T>, int>> unique_points;

        for (int i = 0; i < points.size(); i++) {
            unique_points.push_back(std::make_pair(points[i], 1));
        }

        root = build(unique_points, 0, 0.0, 1.0, 0.0, 1.0);
    }

    // Function to compute mutual information
    double compute_mutual_information() const {
        double mi = 0.0;
        double area = 0.0;

        traverse_and_compute(root.get(), mi, area);

        // int depth = get_tree_depth();

        // int bins_xy = std::pow(2, depth);
        // int bins_x = std::pow(2, depth / 2);
        // int bins_y = std::pow(2, depth / 2);

        // double correction = (bins_xy - 1) / (2. * total_count);

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
    long long total_count;

    Copula * copula;

    double get_bin_area(const KDNode<T> & node) const;

    std::vector<std::pair<Point<T>, int>> count_duplicates_absl(const std::vector<Point<T>>& points) {

        absl::flat_hash_map<std::pair<double, double>, int, absl::Hash<std::pair<double, double>>> point_map;

        point_map.reserve(points.size() / 2);

        for (const auto& pt : points) {
            std::pair<int, int> key = {pt.x, pt.y};
            point_map[key]++;
        }

        std::vector<std::pair<Point<T>, int>> unique_points;
        unique_points.reserve(point_map.size());

        for (const auto& entry : point_map) {
            unique_points.emplace_back(std::make_pair(Point<T>{entry.first.first, entry.first.second}, entry.second));
        }

        return unique_points;
    }

    struct pair_hash {
        std::size_t operator()(const std::pair<double, double>& p) const {
            return std::hash<double>()(p.first) ^ (std::hash<double>()(p.second) << 1);
        }
    };

    std::unique_ptr<KDNode<T>> build(std::vector<std::pair<Point<T>, int>>& points,
                                  int depth,
                                  double min_x, double max_x,
                                  double min_y, double max_y) {

        auto node = std::make_unique<KDNode<T>>();
        node->min_x = min_x;
        node->max_x = max_x;
        node->min_y = min_y;
        node->max_y = max_y;

        if (points.size() <= static_cast<size_t>(max_points)) {
            node->is_leaf = true;
            node->points = points;
            return node;
        }

        int axis = depth % 2;
        node->split_dim = axis;

        if (axis == 0) {
            std::sort(points.begin(), points.end(),
                      [](const std::pair<Point<T>, int>& a, const std::pair<Point<T>, int>& b) -> bool {
                          return a.first.x < b.first.x;
                      });
        } else {
            std::sort(points.begin(), points.end(),
                      [](const std::pair<Point<T>, int>& a, const std::pair<Point<T>, int>& b) -> bool {
                          return a.first.y < b.first.y;
                      });
        }


        size_t median_idx = points.size() / 2;
        double median_val = (axis == 0) ? points[median_idx].first.x
                                        : points[median_idx].first.y;
        node->split_val = median_val;

        std::vector<std::pair<Point<T>, int>> left_points;
        std::vector<std::pair<Point<T>, int>> right_points;

        for (const auto& p : points) {
            double coord = (axis == 0) ? p.first.x : p.first.y;
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

        if (!left_points.empty()) {
            node->left = build(left_points, depth + 1,
                               min_x, (axis == 0 ? median_val : max_x),
                               min_y, (axis == 1 ? median_val : max_y));
        } else {
            auto leaf = std::make_unique<KDNode<T>>();
            leaf->is_leaf = true;
            leaf->min_x = min_x;
            leaf->max_x = (axis == 0 ? median_val : max_x);
            leaf->min_y = min_y;
            leaf->max_y = (axis == 1 ? median_val : max_y);
            node->left = std::move(leaf);
        }

        if (!right_points.empty()) {
            node->right = build(right_points, depth + 1,
                                (axis == 0 ? median_val : min_x), max_x,
                                (axis == 1 ? median_val : min_y), max_y);
        } else {
            auto leaf = std::make_unique<KDNode<T>>();
            leaf->is_leaf = true;
            leaf->min_x = (axis == 0 ? median_val : min_x);
            leaf->max_x = max_x;
            leaf->min_y = (axis == 1 ? median_val : min_y);
            leaf->max_y = max_y;
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
double KDTree<T>::get_bin_area(const KDNode<T> & node) const {
    return node.get_bin_area();
}

template <>
double KDTree<int>::get_bin_area(const KDNode<int> & node) const {

    // Transform to the uniform distribution via the CDF, to get
    // the area in U[0, 1] space

    int x_min = this->copula->cdf_x(node.min_x);
    int x_max = this->copula->cdf_x(node.max_x);

    int y_min = this->copula->cdf_y(node.min_y);
    int y_max = this->copula->cdf_y(node.max_y);

    double width = x_max - x_min;
    double height = x_max - x_min;

    return width * height;
}
