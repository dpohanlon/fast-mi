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

// Define a Point structure with integer coordinates
struct Point {
    int x;
    int y;
};

// Comparator functions for sorting points
bool compareX(const Point& a, const Point& b) {
    return a.x < b.x;
}

bool compareY(const Point& a, const Point& b) {
    return a.y < b.y;
}

// kd-tree node
struct KDNode {
    // Bounding box for the node (useful for range queries)
    int min_x, max_x;
    int min_y, max_y;

    // If leaf node, store points and their counts
    std::vector<std::pair<Point, int>> points; // Pair of Point and count

    // If internal node, store the splitting dimension and splitting value
    bool is_leaf;
    int split_dim; // 0 for x, 1 for y
    int split_val;

    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode() : is_leaf(false), split_dim(0), split_val(0),
               min_x(INT32_MAX), max_x(INT32_MIN),
               min_y(INT32_MAX), max_y(INT32_MIN) {}
};

// kd-tree class
class KDTree {
public:
    KDTree(const std::vector<Point>& points, int max_points_per_leaf = 10)
        : max_points(max_points_per_leaf) {
        // Preprocess points to count duplicates
        std::vector<std::pair<Point, int>> unique_points = count_duplicates_unordered_map(points);
        root = build(unique_points, 0);
        total_count = compute_total_count(root.get());
    }

    // Function to compute mutual information
    double compute_mutual_information(const std::function<double(int)>& p_x_func,
                                     const std::function<double(int)>& p_y_func) const {
        double mi = 0.0;
        traverse_and_compute(root.get(), p_x_func, p_y_func, mi);
        return mi;
    }

private:
    std::unique_ptr<KDNode> root;
    int max_points; // Maximum points per leaf
    long long total_count; // Total number of points

    // Function to count duplicates and return unique points with their counts
    std::vector<std::pair<Point, int>> count_duplicates(const std::vector<Point>& points) {
        std::vector<Point> sorted_points = points;
        std::sort(sorted_points.begin(), sorted_points.end(),
                  [](const Point& a, const Point& b) -> bool {
                      if (a.x != b.x)
                          return a.x < b.x;
                      return a.y < b.y;
                  });

        std::vector<std::pair<Point, int>> unique_points;
        if (sorted_points.empty()) return unique_points;

        Point current = sorted_points[0];
        int count = 1;

        #pragma omp simd
        for (size_t i = 1; i < sorted_points.size(); ++i) {
            if (sorted_points[i].x == current.x && sorted_points[i].y == current.y) {
                count++;
            } else {
                unique_points.emplace_back(std::make_pair(current, count));
                current = sorted_points[i];
                count = 1;
            }
        }
        unique_points.emplace_back(std::make_pair(current, count));
        return unique_points;
    }

    std::vector<std::pair<Point, int>> count_duplicates_absl(const std::vector<Point>& points) {
        // Define the flat_hash_map with pair<int, int> as key
        absl::flat_hash_map<std::pair<int, int>, int, absl::Hash<std::pair<int, int>>> point_map;

        // Reserve space to minimize rehashing (optional but recommended)
        point_map.reserve(points.size() / 2); // Adjust based on expected uniqueness

        // Count occurrences
        for (const auto& pt : points) {
            std::pair<int, int> key = {pt.x, pt.y};
            point_map[key]++;
        }

        // Convert the map to a vector of unique points with counts
        std::vector<std::pair<Point, int>> unique_points;
        unique_points.reserve(point_map.size());

        for (const auto& entry : point_map) {
            unique_points.emplace_back(std::make_pair(Point{entry.first.first, entry.first.second}, entry.second));
        }

        return unique_points;
    }

    struct pair_hash {
        std::size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    // Function to count duplicates using std::unordered_map
    std::vector<std::pair<Point, int>> count_duplicates_unordered_map(const std::vector<Point>& points) {
        // Define the unordered_map with std::pair<int, int> as key
        std::unordered_map<std::pair<int, int>, int, pair_hash> point_map;

        // Reserve space to minimize rehashing (optional but recommended)
        point_map.reserve(points.size() / 2); // Adjust based on expected uniqueness

        // Count occurrences
        for (const auto& pt : points) {
            std::pair<int, int> key = {pt.x, pt.y};
            point_map[key]++;
        }

        // Convert the map to a vector of unique points with counts
        std::vector<std::pair<Point, int>> unique_points;
        unique_points.reserve(point_map.size());

        for (const auto& entry : point_map) {
            unique_points.emplace_back(std::make_pair(Point{entry.first.first, entry.first.second}, entry.second));
        }

        return unique_points;
    }

    // Recursive build function with early stopping
    std::unique_ptr<KDNode> build(std::vector<std::pair<Point, int>> points, int depth) {
        if (points.empty()) return nullptr;

        auto node = std::make_unique<KDNode>();

        // Update bounding box
        for (const auto& p : points) {
            node->min_x = std::min(node->min_x, p.first.x);
            node->max_x = std::max(node->max_x, p.first.x);
            node->min_y = std::min(node->min_y, p.first.y);
            node->max_y = std::max(node->max_y, p.first.y);
        }

        // Check stopping criteria
        if (points.size() <= static_cast<size_t>(max_points)) {
            node->is_leaf = true;
            node->points = std::move(points);
            return node;
        }

        // Determine splitting dimension: alternate between x and y
        int axis = depth % 2;
        node->split_dim = axis;

        // Sort points based on the splitting axis
        if (axis == 0) {
            std::sort(points.begin(), points.end(),
                      [](const std::pair<Point, int>& a, const std::pair<Point, int>& b) -> bool {
                          return a.first.x < b.first.x;
                      });
        } else {
            std::sort(points.begin(), points.end(),
                      [](const std::pair<Point, int>& a, const std::pair<Point, int>& b) -> bool {
                          return a.first.y < b.first.y;
                      });
        }

        // Find median
        size_t median_idx = points.size() / 2;
        int median_val = (axis == 0) ? points[median_idx].first.x : points[median_idx].first.y;
        node->split_val = median_val;

        // Partition points into left and right subsets
        std::vector<std::pair<Point, int>> left_points;
        std::vector<std::pair<Point, int>> right_points;

        for (const auto& p : points) {
            int coord = (axis == 0) ? p.first.x : p.first.y;
            if (coord < median_val) {
                left_points.emplace_back(p);
            } else if (coord > median_val) {
                right_points.emplace_back(p);
            } else {
                // If coordinate equals median, distribute to balance the tree
                // Simple strategy: alternate assignment
                if (left_points.size() <= right_points.size()) {
                    left_points.emplace_back(p);
                } else {
                    right_points.emplace_back(p);
                }
            }
        }

        // Recursively build left and right subtrees
        node->left = build(left_points, depth + 1);
        node->right = build(right_points, depth + 1);

        return node;
    }

    // Function to compute total count of points
    long long compute_total_count(const KDNode* node) const {
        if (!node) return 0;
        if (node->is_leaf) {
            long long sum = 0;
            for (const auto& p : node->points) {
                sum += p.second;
            }
            return sum;
        }
        return compute_total_count(node->left.get()) + compute_total_count(node->right.get());
    }

    // Can I make the underlying storage here an eigen vector, and then just push it through the NB calculation? Or maybe even populate it with points and the corresponding NB beforehand? -> Take the two Eigen vectors, calculate the NB, and then pop the points with (x, y, nb_x, nb_y)

    // Take either a function p_x, p_y, or vectors p_x, p_y

    // Function to traverse the tree and compute mutual information
    void traverse_and_compute(const KDNode* node,
                               const std::function<double(int)>& p_x_func,
                               const std::function<double(int)>& p_y_func,
                               double& mi) const {
        if (!node) return;

        if (node->is_leaf) {
            for (const auto& p : node->points) {
                double p_xy = static_cast<double>(p.second) / static_cast<double>(total_count);
                double p_x = p_x_func(p.first.x);
                double p_y = p_y_func(p.first.y);

                if (p_xy > 0 && p_x > 0 && p_y > 0) {
                    mi += p_xy * std::log(p_xy / (p_x * p_y));
                }
            }
            return;
        }

        // Recurse on left and right children
        traverse_and_compute(node->left.get(), p_x_func, p_y_func, mi);
        traverse_and_compute(node->right.get(), p_x_func, p_y_func, mi);
    }
};
