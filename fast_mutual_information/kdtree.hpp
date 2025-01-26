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

struct Point {
    double x;
    double y;
};

// kd-tree node
struct KDNode {
    // Bounding box for the node (useful for range queries)
    double min_x, max_x;
    double min_y, max_y;

    // If leaf node, store points and their counts
    std::vector<std::pair<Point, int>> points; // Pair of Point and count

    // If internal node, store the splitting dimension and splitting value
    bool is_leaf;
    int split_dim; // 0 for x, 1 for y
    double split_val;

    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode() : is_leaf(false), split_dim(0), split_val(0),
               min_x(1E8), max_x(-1E8),
               min_y(1E8), max_y(-1E8) {}
               // min_x(1.0), max_x(0.0),
               // min_y(1.0), max_y(0.0) {}

    double get_bin_area() const {
        double width = this->max_x - this->min_x;
        double height = this->max_y - this->min_y;
        return width * height;
    }

};

// kd-tree class
class KDTree {
public:

    KDTree() {}

    KDTree(const std::vector<Point>& points, Copula * copula, int max_points_per_leaf = 10)
        : max_points(max_points_per_leaf), copula(copula) {

        // total_count = points.size(); // Or # leaves?

        long long sum_points = 0;
        for (auto &p : points) sum_points += 1; // or if using duplicates, sum up p.second
        total_count = sum_points;

        // Preprocess points to count duplicates

        // For ints!

        // std::vector<std::pair<Point, int>> unique_points = count_duplicates_unordered_map(points);

        // std::cout << "Getting points" << std::endl;

        std::vector<std::pair<Point, int>> unique_points;

        for (int i = 0; i < points.size(); i++) {
            unique_points.push_back(std::make_pair(points[i], 1));
        }

        // for (auto p : unique_points) {
        //     std::cout << p.first.x << " " << p.first.y << " " << p.second << std::endl;
        // }

        // std::cout << "Building" << std::endl;

        // root = build(unique_points, 0);
        root = build(unique_points, 0, 0.0, 1.0, 0.0, 1.0);
        // build(unique_points, 0, 0.0, 1.0, 0.0, 1.0);
        // total_count = compute_total_count(root.get());
    }

    // Function to compute mutual information
    double compute_mutual_information() const {
        double mi = 0.0;
        double pxy = 0.0;

        traverse_and_compute(root.get(), mi, pxy);

        int depth = get_tree_depth();

        std::cout << "depth " << depth << std::endl;

        int bins_xy = std::pow(2, depth);
        int bins_x = std::pow(2, depth / 2);
        int bins_y = std::pow(2, depth / 2);

        // double correction = (1./(2. * total_count)) * ((bins_x - 1) + (bins_y - 1) - (bins_xy - 1));

        // double correction = (1./(2. * total_count)) * ( - (bins_xy - 1));

        double correction = (bins_xy - 1) / (2. * total_count);

        std::cout << "pxy " << pxy << std::endl;
        return mi;
    }

    int calculate_depth(const KDNode* node) const {
        if (!node) return 0; // Base case: empty node
        if (node->is_leaf) return 1; // Leaf nodes contribute a depth of 1

        // Recurse on left and right children and return the maximum depth
        int left_depth = calculate_depth(node->left.get());
        int right_depth = calculate_depth(node->right.get());

        return 1 + std::max(left_depth, right_depth);
    }

    // Wrapper function to calculate depth from the root
    int get_tree_depth() const {
        return calculate_depth(root.get());
    }

private:
    std::unique_ptr<KDNode> root;
    int max_points; // Maximum points per leaf
    long long total_count; // Total number of points

    Copula * copula;

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
        absl::flat_hash_map<std::pair<double, double>, int, absl::Hash<std::pair<double, double>>> point_map;

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
        std::size_t operator()(const std::pair<double, double>& p) const {
            return std::hash<double>()(p.first) ^ (std::hash<double>()(p.second) << 1);
        }
    };

    // Function to count duplicates using std::unordered_map
    std::vector<std::pair<Point, int>> count_duplicates_unordered_map(const std::vector<Point>& points) {
        // Define the unordered_map with std::pair<int, int> as key
        std::unordered_map<std::pair<double, double>, int, pair_hash> point_map;

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

    std::unique_ptr<KDNode> build(std::vector<std::pair<Point, int>>& points,
                                  int depth,
                                  double min_x, double max_x,
                                  double min_y, double max_y) {

        auto node = std::make_unique<KDNode>();
        node->min_x = min_x;
        node->max_x = max_x;
        node->min_y = min_y;
        node->max_y = max_y;

        // Check stopping criteria
        if (points.size() <= static_cast<size_t>(max_points)) {
            node->is_leaf = true;
            node->points = points;
            return node;
        }

        // Determine splitting dimension: 0 for x, 1 for y
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
        double median_val = (axis == 0) ? points[median_idx].first.x
                                        : points[median_idx].first.y;
        node->split_val = median_val;

        // Partition points into left and right subsets
        std::vector<std::pair<Point, int>> left_points;
        std::vector<std::pair<Point, int>> right_points;

        for (const auto& p : points) {
            double coord = (axis == 0) ? p.first.x : p.first.y;
            if (coord < median_val) {
                left_points.emplace_back(p);
            } else if (coord > median_val) {
                right_points.emplace_back(p);
            } else {
                // If coordinate equals median, distribute to balance the tree
                if (left_points.size() <= right_points.size()) {
                    left_points.emplace_back(p);
                } else {
                    right_points.emplace_back(p);
                }
            }
        }

        // Handle potential empty subsets by enforcing the bounding box split
        // Build left child
        if (!left_points.empty()) {
            node->left = build(left_points, depth + 1,
                               min_x, (axis == 0 ? median_val : max_x),
                               min_y, (axis == 1 ? median_val : max_y));
        } else {
            // Create an empty leaf for that region
            auto leaf = std::make_unique<KDNode>();
            leaf->is_leaf = true;
            leaf->min_x = min_x;
            leaf->max_x = (axis == 0 ? median_val : max_x);
            leaf->min_y = min_y;
            leaf->max_y = (axis == 1 ? median_val : max_y);
            node->left = std::move(leaf);
        }

        // Build right child
        if (!right_points.empty()) {
            node->right = build(right_points, depth + 1,
                                (axis == 0 ? median_val : min_x), max_x,
                                (axis == 1 ? median_val : min_y), max_y);
        } else {
            // Create an empty leaf for that region
            auto leaf = std::make_unique<KDNode>();
            leaf->is_leaf = true;
            leaf->min_x = (axis == 0 ? median_val : min_x);
            leaf->max_x = max_x;
            leaf->min_y = (axis == 1 ? median_val : min_y);
            leaf->max_y = max_y;
            node->right = std::move(leaf);
        }

        return node;
    }

    // // // Recursive build function with early stopping
    // std::unique_ptr<KDNode> build(std::vector<std::pair<Point, int>> points, int depth) {

    //     // std::cout << depth << std::endl;

    //     if (points.empty()) return nullptr;

    //     auto node = std::make_unique<KDNode>();

    //     // std::cout << "Updating box" << std::endl;

    //     // Update bounding box
    //     for (const auto& p : points) {
    //         node->min_x = std::min(node->min_x, p.first.x);
    //         node->max_x = std::max(node->max_x, p.first.x);
    //         node->min_y = std::min(node->min_y, p.first.y);
    //         node->max_y = std::max(node->max_y, p.first.y);
    //     }

    //     // std::cout << "Stopping? " << points.size() << " " << max_points << std::endl;
    //     //

    //     // Check stopping criteria
    //     if (points.size() <= static_cast<size_t>(max_points)) {
    //         node->is_leaf = true;
    //         node->points = std::move(points);
    //         // if (node->points.size() < max_points) std::cout << "stopping " << node->points.size() << std::endl;
    //         return node;
    //     }

    //     // std::cout << "Stopping? No" << std::endl;

    //     // Determine splitting dimension: alternate between x and y
    //     int axis = depth % 2;
    //     node->split_dim = axis;

    //     // These should already be sorted?

    //     // Sort points based on the splitting axis
    //     if (axis == 0) {
    //         std::sort(points.begin(), points.end(),
    //                   [](const std::pair<Point, int>& a, const std::pair<Point, int>& b) -> bool {
    //                       return a.first.x < b.first.x;
    //                   });
    //     } else {
    //         std::sort(points.begin(), points.end(),
    //                   [](const std::pair<Point, int>& a, const std::pair<Point, int>& b) -> bool {
    //                       return a.first.y < b.first.y;
    //                   });
    //     }

    //     // Find median
    //     size_t median_idx = points.size() / 2;
    //     double median_val = (axis == 0) ? points[median_idx].first.x : points[median_idx].first.y;
    //     // std::cout << "Med idx " << median_idx << " val " << median_val << std::endl;
    //     node->split_val = median_val;

    //     // Partition points into left and right subsets
    //     std::vector<std::pair<Point, int>> left_points;
    //     std::vector<std::pair<Point, int>> right_points;

    //     // std::cout << "Populating " << median_val << std::endl;

    //     for (const auto& p : points) {
    //         // Int for discrete!
    //         double coord = (axis == 0) ? p.first.x : p.first.y;
    //         // std::cout << "med: " << median_val << " coord: " << coord << std::endl;
    //         if (coord < median_val) {
    //             left_points.emplace_back(p);
    //         } else if (coord > median_val) {
    //             right_points.emplace_back(p);
    //         } else {
    //             // If coordinate equals median, distribute to balance the tree
    //             // Simple strategy: alternate assignment
    //             if (left_points.size() <= right_points.size()) {
    //                 left_points.emplace_back(p);
    //             } else {
    //                 right_points.emplace_back(p);
    //             }
    //         }
    //     }

    //     // std::cout << "L " << left_points.size() << " R " << right_points.size() << std::endl;

    //     // exit(0);

    //     // Recursively build left and right subtrees
    //     node->left = build(left_points, depth + 1);
    //     node->right = build(right_points, depth + 1);

    //     return node;
    // }

    // Function to compute total count of points
    // long long compute_total_count(const KDNode* node) const {
    //     if (!node) return 0;
    //     if (node->is_leaf) {
    //         long long sum = 0;
    //         for (const auto& p : node->points) {
    //             sum += p.second;
    //         }
    //         return sum;
    //     }
    //     return compute_total_count(node->left.get()) + compute_total_count(node->right.get());
    // }
    long long compute_total_count(const KDNode* node) const {
        if (!node) return 0;
        if (node->is_leaf) {
            return 1;
        }
        return compute_total_count(node->left.get()) + compute_total_count(node->right.get());
    }

    // Can I make the underlying storage here an eigen vector, and then just push it through the NB calculation? Or maybe even populate it with points and the corresponding NB beforehand? -> Take the two Eigen vectors, calculate the NB, and then pop the points with (x, y, nb_x, nb_y)
    //

    double calculate_p_xy(int count) const {
        return static_cast<double>(count) / static_cast<double>(total_count);
    }

    double calculate_p_xy(int count, double bin_area) const {
        return static_cast<double>(count) / (static_cast<double>(total_count) * bin_area);
    }


    // std::pair<double, double> calculate_marginal_probs(const Point& point) const {

    //     // std::cout << point.x << " " << point.y << std::endl;

    //     // Transform to uniform distribution using the CDF
    //     double u = copula->cdf_x(point.x);
    //     // std::cout << "u " << u << std::endl;

    //     double v = copula->cdf_y(point.y);
    //     // std::cout << "v " << v << std::endl;

    //     // Transform back using the inverse CDF of the desired marginals
    //     double p_x = copula->p_x(copula->icdf_x(u));
    //     double p_y = copula->p_y(copula->icdf_y(v));

    //     return std::make_pair(p_x, p_y);
    // }

    std::pair<double, double> calculate_marginal_probs(const Point& point) const {
        return std::make_pair(1.0, 1.0);
    }

    void traverse_and_compute(const KDNode* node, double& mi, double &pxy) const {
        if (!node) return;

        if (node->is_leaf) {

            // std::cout << "x: " << node->min_x << " " << node->max_x << " \n";
            // std::cout << "y: " << node->min_y << " " << node->max_y << std::endl;
            // std::cout << std::endl;

            // 1. Count points in the bin (consider duplicates)
            int bin_count = node->points.size(); // Adjust for duplicates if needed
            // std::cout << bin_count << std::endl;

            double bin_area = node->get_bin_area();

            // std::cout << "area " << bin_area << std::endl;

            // 2. Calculate p_xy for the bin
            double p_xy = calculate_p_xy(bin_count, bin_area);

            // std::cout << p_xy << std::endl;

            // 4. Calculate MI contribution for the bin

            // Add p_x, p_y for generality?
            //
            double w_x = node->max_x - node->min_x;
            double w_y = node->max_y - node->min_y;

            if (p_xy > 0) {
                // std::cout << p_xy << " " << std::log(p_xy) << std::endl;
                // p_x_bin and p_y_bin are both 1.0 due to the copula transformation
                // mi += p_xy * std::log(p_xy / (w_x * w_y)) * bin_area;  // Add epsilon for numerical stability
                mi += p_xy * std::log(p_xy) * bin_area;  // Add epsilon for numerical stability
            }

            pxy += bin_area;

            // int bin_count = node->points.size();
            // double bin_prob = static_cast<double>(bin_count) / static_cast<double>(total_count);
            // double w_x = node->max_x - node->min_x;
            // double w_y = node->max_y - node->min_y;

            // if (bin_prob > 0) {
            //     mi += bin_prob * std::log(bin_prob / (w_x * w_y));
            // }
            // pxy += bin_prob;

            return;
        }



        // Recurse on left and right children
        traverse_and_compute(node->left.get(), mi, pxy);
        traverse_and_compute(node->right.get(), mi, pxy);
    }

};

//     void traverse_and_compute(const KDNode* node,
//                             double& mi,
//                             double& pxy_accum) const
//     {
//         if (!node) return;

//         if (node->is_leaf) {
//             // Sum up total points in this bin
//             int bin_count = 0;
//             for (auto &pr : node->points) {
//                 bin_count += pr.second;
//             }
//             double bin_area = node->get_bin_area();

//             // p_xy approximate within that bin
//             double p_xy_bin = (double)bin_count / (total_count * bin_area);

//             // Get average p_x and p_y inside this bin
//             double p_x_bin = 0.0;
//             double p_y_bin = 0.0;
//             for (auto &pr : node->points) {
//                 // pr.first.x, pr.first.y
//                 p_x_bin += copula->p_x(pr.first.x) * pr.second;
//                 p_y_bin += copula->p_y(pr.first.y) * pr.second;
//             }
//             p_x_bin /= bin_count;
//             p_y_bin /= bin_count;

//             // Add the local contribution to MI:
//             // p_xy_bin * log( p_xy_bin / [p_x_bin * p_y_bin] ) * bin_area
//             if (p_xy_bin > 1e-14 && p_x_bin > 1e-14 && p_y_bin > 1e-14) {
//                 mi += p_xy_bin * std::log(p_xy_bin / (p_x_bin * p_y_bin)) * bin_area;
//             }

//             pxy_accum += p_xy_bin * bin_area;
//             return;
//         }

//         traverse_and_compute(node->left.get(), mi, pxy_accum);
//         traverse_and_compute(node->right.get(), mi, pxy_accum);
//     }
// };
