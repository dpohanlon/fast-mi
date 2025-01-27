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
#include "point.hpp"
#include "utils.hpp"


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

        // std::cout << area << std::endl;

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

    std::vector<std::pair<Point<int>, int>> count_duplicates_absl(const std::vector<Point<int>>& points) {

        absl::flat_hash_map<std::pair<int, int>, int, absl::Hash<std::pair<int, int>>> point_map;

        point_map.reserve(points.size() / 2);

        for (const auto& pt : points) {
            std::pair<int, int> key = {pt.x, pt.y};
            point_map[key]++;
        }

        std::vector<std::pair<Point<int>, int>> unique_points;
        unique_points.reserve(point_map.size());

        for (const auto& entry : point_map) {
            unique_points.emplace_back(std::make_pair(Point<T>{entry.first.first, entry.first.second}, entry.second));
        }

        return unique_points;
    }

    std::unique_ptr<KDNode<T>> build(std::vector<std::pair<Point<T>, int>>& points,
                                  int depth,
                                  T min_x, T max_x,
                                  T min_y, T max_y) {

        auto node = std::make_unique<KDNode<T>>();
        node->min_x = min_x;
        node->max_x = max_x;
        node->min_y = min_y;
        node->max_y = max_y;

        bool degenerate_split = (max_x - min_x < 1E-8) || (max_y - min_y < 1E-8);

        // This is controlled by the number of points rather than the number of points including the duplicates as we don't want to end up with nowhere to split
        if (points.size() <= static_cast<size_t>(max_points) || degenerate_split) {
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

        // if (!left_points.empty()) {
         if (!left_points.empty()) {
            // if (left_points.size() == 1) {
            //     std::cout << "Left size 1\n";
            // }
            // std::cout << "left size " << left_points.size() << std::endl;
            node->left = build(left_points, depth + 1,
                               min_x, (axis == 0 ? median_val : max_x),
                               min_y, (axis == 1 ? median_val : max_y));
        } else {
            std::cout << "Left empty\n";
            auto leaf = std::make_unique<KDNode<T>>();
            leaf->is_leaf = true;
            leaf->min_x = min_x;
            leaf->max_x = (axis == 0 ? median_val : max_x);
            leaf->min_y = min_y;
            leaf->max_y = (axis == 1 ? median_val : max_y);
            node->left = std::move(leaf);
        }

        if (!right_points.empty()) {
            // if (left_points.size() == 1) {
                // std::cout << "Right size 1\n";
            // }
            // std::cout << "right size " << right_points.size() << std::endl;
            node->right = build(right_points, depth + 1,
                                (axis == 0 ? median_val : min_x), max_x,
                                (axis == 1 ? median_val : min_y), max_y);
        } else {
            std::cout << "Right empty\n";
            auto leaf = std::make_unique<KDNode<T>>();
            leaf->is_leaf = true;
            leaf->min_x = (axis == 0 ? median_val : min_x);
            leaf->max_x = max_x;
            leaf->min_y = (axis == 1 ? median_val : min_y);
            leaf->max_y = max_y;
            node->right = std::move(leaf);
        }

        // degenerate_split = (node->max_x - node->min_x < 1E-8) || (node->max_y - node->min_y < 1E-8);

        // bool degenerate_left = (node->left) && ( (node->max_x - node->min_x < 1E-8) || (node->max_y - node->min_y < 1E-8));

        // bool degenerate_right = (node->right) && ( (node->max_x - node->min_x < 1E-8) || (node->max_y - node->min_y < 1E-8));

        // if (degenerate_split) std:: cout << "DEGENERACY" << std::endl;
        // if (degenerate_left) std:: cout << "DEGENERACY L" << std::endl;
        // if (degenerate_right) std:: cout << "DEGENERACY R" << std::endl;

        // std::cout << "THIS " << node->min_x << " " << node->max_x << " " << node->min_y << " " << node->max_y << std::endl;

        // if (node->left) {
        //     std::cout << "LEFT " << node->left->min_x << " " << node->left->max_x << " " << node->left->min_y << " " << node->left->max_y << std::endl;
        //     if (node->left->min_y == node->left->max_y) {
        //         std::cout << node->left->points.size() << std::endl;
        //         std::cout << std::endl;
        //         for (auto p : node->left->points) {
        //             std::cout << p.second << " " << p.first.x << " " << p.first.y << std::endl;
        //         }
        //         // exit(0);
        //     }
        // }

        // if (node->right) {
        //     std::cout << "RIGHT " << node->right->min_x << " " << node->right->max_x << " " << node->right->min_y << " " << node->right->max_y << std::endl;
        // }

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

            // std::cout << bin_count << " " << bin_area << " " << p_xy << std::endl;

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

    root = build(unique_points, 0, 0.0, 1.0, 0.0, 1.0);
}

template <>
KDTree<int>::KDTree(const std::vector<Point<int>>& points, Copula * copula, int max_points_per_leaf)
    : max_points(max_points_per_leaf), copula(copula) {

    total_count = points.size();

    std::vector<std::pair<Point<int>, int>> unique_points = count_duplicates_absl(points);

    // These are the boundaries of the input data that then get mapped to [0, 1, 0, 1] when transformed via the CDF

    auto [min_x, max_x, min_y, max_y] = get_bounds(points);

    root = build(unique_points, 0, min_x, max_x, min_y, max_y);
}

template <typename T>
double KDTree<T>::get_bin_area(const KDNode<T> & node) const {
    return node.get_bin_area();
}

template <>
double KDTree<int>::get_bin_area(const KDNode<int> & node) const {

    // Transform to the uniform distribution via the CDF, to get
    // the area in U[0, 1] space

    double x_min = this->copula->cdf_x(node.min_x);
    double x_max = this->copula->cdf_x(node.max_x);

    double y_min = this->copula->cdf_y(node.min_y);
    double y_max = this->copula->cdf_y(node.max_y);

    // std::cout << node.min_x << " " << node.max_x << " " << node.min_y << " " << node.max_y << std::endl;
    // std::cout << x_min << " " << x_max << " " << y_min << " " << y_max << std::endl;
    // std::cout << std::endl;

    double width = x_max - x_min;
    double height = y_max - y_min;

    return width * height;

}
