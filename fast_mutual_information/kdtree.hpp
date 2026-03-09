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
#include <type_traits>

#include "copula.hpp"
#include "point.hpp"
#include "utils.hpp"

// #define DEBUG

enum class MiMode { Copula, Raw };

template <typename T>
struct KDNode {

    Bounds<T> bounds;

    std::vector<std::pair<Point<T>, int>> points;

    size_t begin_idx;
    size_t end_idx;    // one-past-the-last in points_storage

    bool is_leaf;
    int split_dim;  // 0 for x, 1 for y
    T split_val;
    int leaf_id = -1;

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

    // No copula, defaults to MiMode::Raw
    KDTree(const std::vector<Point<T>>& points,
           int min_points_per_leaf = 10, bool zi = false);

    KDTree(std::vector<std::pair<Point<int>, int>>& unique_points, int nPoints,
           Bounds<int> bounds, Copula<int>* copula, int min_points_per_leaf, bool zi = false);

    template<class InputIt>
    KDTree(InputIt first, InputIt last,
           Copula<int>* copula,
           int min_points_per_leaf = 10,
           bool zi = false);

    MiMode mode_ = MiMode::Copula;

    void set_mode(MiMode m) { mode_ = m; }

    double get_correction() const {
        int depth = get_tree_depth();

        int bins_xy = std::pow(2, depth);
        int bins_x = std::pow(2, depth / 2);
        int bins_y = std::pow(2, depth / 2);

        return (bins_xy - 1) / (2. * total_count);
    }

    std::pair<double, double> compute_mutual_information() const {
        double mi = 0.0, area = 0.0, chi2 = 0.0;
        const double root_area_u = compute_root_area_u();
        traverse_and_compute(root.get(), root_area_u, mi, area, chi2);
        return {mi, chi2};
    }

    int calculate_depth(const KDNode<T>* node) const {
        if (!node) return 0;
        if (node->is_leaf) return 1;

        int left_depth = calculate_depth(node->left.get());
        int right_depth = calculate_depth(node->right.get());

        return 1 + std::max(left_depth, right_depth);
    }

    template<class F>
    void for_each_point(F&& emit) const;

    int get_tree_depth() const { return calculate_depth(root.get()); }

    void dumpSplittingValuesToCSV(const std::string &filename) const;

    template<class InputIt>
    std::pair<double, int> chi2_holdout(InputIt first, InputIt last, int n_test, double min_expected = 5.0) const;

   private:
    std::unique_ptr<KDNode<T>> root;
    int min_points;
    int total_count;
    bool zi;

    std::vector<std::pair<Point<T>,int>> points_storage;

    Copula<T>* copula;

    double get_bin_area(const KDNode<T>& node) const;

    template<class F>
    void for_each_point_impl(const KDNode<int>* node, F&& emit) const;

    void dumpSplittingValuesHelper(const KDNode<T>* node, std::ofstream &out, int depth) const;

    double compute_root_area_u() const {
        // if (mode_ == MiMode::Raw) return 1.0;
        // if constexpr (std::is_integral<T>::value) {
        //     const auto& b = root->bounds;
        //     const double x_lo = (b.min_x > std::numeric_limits<int>::min())
        //                             ? copula->cdf_x(b.min_x - 1) : 0.0;
        //     const double x_hi = copula->cdf_x(b.max_x);
        //     const double y_lo = (b.min_y > std::numeric_limits<int>::min())
        //                             ? copula->cdf_y(b.min_y - 1) : 0.0;
        //     const double y_hi = copula->cdf_y(b.max_y);
        //     return std::max(1e-15, (x_hi - x_lo) * (y_hi - y_lo));
        // } else {
        //     return 1.0;
        // }
        return 1.0;
    }

    // For ints this can be optimised by sorting!

    std::vector<std::pair<Point<int>, int>> count_duplicates_unordered(
        const std::vector<Point<int>>& points) {
        std::unordered_map<Point<int>, int> counts;
        // ska::flat_hash_map<Point<int>, int> counts;
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

    mutable bool leaf_cache_ready = false;
    mutable std::vector<const KDNode<T>*> leaf_nodes;
    mutable std::vector<double> leaf_prob;

    void build_leaf_cache() const {
        leaf_nodes.clear();
        leaf_prob.clear();

        std::vector<const KDNode<T>*> stack;
        stack.push_back(root.get());

        while (!stack.empty()) {
            const KDNode<T>* node = stack.back();
            stack.pop_back();
            if (!node) continue;

            if (node->is_leaf) {
                const int id = static_cast<int>(leaf_nodes.size());
                const_cast<KDNode<T>*>(node)->leaf_id = id;
                leaf_nodes.push_back(node);
                leaf_prob.push_back(leaf_rect_null_mass(*node));
            } else {
                stack.push_back(node->right.get());
                stack.push_back(node->left.get());
            }
        }

        double Z = 0.0;
        for (double a : leaf_prob) Z += a;
        if (Z <= 0.0) {
            for (auto& p : leaf_prob) p = 0.0;
            leaf_cache_ready = true;
            return;
        }
        for (auto& p : leaf_prob) p /= Z;

        leaf_cache_ready = true;
    }

    int locate_leaf_id(const Point<T>& p) const {
        const KDNode<T>* node = root.get();
        while (node && !node->is_leaf) {
            const T coord = (node->split_dim == 0) ? p.x : p.y;
            node = (coord < node->split_val) ? node->left.get() : node->right.get();
        }
        return node ? node->leaf_id : -1;
    }

    // Replace your count_duplicates_unordered with this:
    std::vector<std::pair<Point<int>, int>> count_duplicates_hist(
        const std::vector<Point<int>>& points,
        int min_x, int max_x,
        int min_y, int max_y)
    {
        const int nx = max_x - min_x + 1;
        const int ny = max_y - min_y + 1;

        // flat histogram + touched‐bins list
        std::vector<int> hist(nx * ny, 0);
        std::vector<int> touched;
        touched.reserve(points.size());

        // 1) fill, tracking which bins go from 0→1
        for (const auto& pt : points) {
            int ix  = pt.x - min_x;
            int iy  = pt.y - min_y;
            int idx = ix * ny + iy;
            if (hist[idx]++ == 0) {
                touched.push_back(idx);
            }
        }

        // 2) collect non‑zero counts
        std::vector<std::pair<Point<int>, int>> result;
        result.reserve(touched.size());
        for (int idx : touched) {
            int ix = (idx / ny) + min_x;
            int iy = (idx % ny) + min_y;
            result.emplace_back(Point<int>{ix, iy}, hist[idx]);
        }

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

    long long range_mass(size_t b, size_t e) const {
        long long s = 0;
        for (size_t i = b; i < e; ++i) s += (long long)points_storage[i].second;
        return s;
    }

    bool is_degenerate_bounds(const Bounds<T>& bounds) const {
        if constexpr (std::is_integral<T>::value) {
            return (bounds.min_x >= bounds.max_x) || (bounds.min_y >= bounds.max_y);
        } else {
            return (bounds.max_x - bounds.min_x < 1e-8) || (bounds.max_y - bounds.min_y < 1e-8);
        }
    }

    static inline T coord_of(const std::pair<Point<T>, int>& pr, int axis) {
        return (axis == 0) ? pr.first.x : pr.first.y;
    }

    void sort_range_by_axis(size_t b, size_t e, int axis) {
        auto comp = [axis](const auto& A, const auto& B) {
            return coord_of(A, axis) < coord_of(B, axis);
        };
        std::sort(points_storage.begin() + b, points_storage.begin() + e, comp);
    }

    T first_coord(size_t b, int axis) const {
        return coord_of(points_storage[b], axis);
    }

    T last_coord(size_t e, int axis) const {
        return coord_of(points_storage[e - 1], axis);
    }

    T first_coord_greater_than(size_t b, size_t e, int axis, T v0) const {
        for (size_t i = b; i < e; ++i) {
            const T v = coord_of(points_storage[i], axis);
            if (v > v0) return v;
        }
        return v0; // caller must check v0 != max_coord first
    }

    T choose_integer_cut_weighted(
        size_t b, size_t e,
        int axis,
        long long mass
    ) const {
        // Assumes [b,e) is already sorted by coord(axis)
        const T min_c = first_coord(b, axis);
        const long long target = (mass + 1) / 2;

        long long cum = 0;

        // Iterate by blocks of equal coordinate
        for (size_t i = b; i < e; ) {
            const T v = coord_of(points_storage[i], axis);

            long long block_mass = 0;
            size_t j = i;
            while (j < e && coord_of(points_storage[j], axis) == v) {
                block_mass += (long long)points_storage[j].second;
                ++j;
            }

            // cut = v means left is coord < v, so require v > min_c to ensure non-empty left.
            if (cum + block_mass >= target && v > min_c) {
                return v;
            }

            cum += block_mass;
            i = j;
        }

        // If we never crossed target at a v>min_c, fall back to the first coord > min_c.
        // Caller should have checked min_c != max_c, so this exists.
        return first_coord_greater_than(b, e, axis, min_c);
    }

    size_t lower_bound_coord(size_t b, size_t e, int axis, T cut) const {
        // Assumes [b,e) sorted by coord(axis). Returns first idx with coord >= cut.
        size_t i = b;
        while (i < e && coord_of(points_storage[i], axis) < cut) ++i;
        return i;
    }

    void carve_child_bounds(
        const Bounds<T>& parent,
        int axis,
        T cut,
        Bounds<T>& L,
        Bounds<T>& R
    ) const {
        L = parent;
        R = parent;

        if (axis == 0) {
            if constexpr (std::is_integral<T>::value) {
                L.max_x = cut - 1;
                R.min_x = cut;
            } else {
                L.max_x = cut;
                R.min_x = cut;
            }
        } else {
            if constexpr (std::is_integral<T>::value) {
                L.max_y = cut - 1;
                R.min_y = cut;
            } else {
                L.max_y = cut;
                R.min_y = cut;
            }
        }
    }

    void make_leaf(KDNode<T>* node, size_t b, size_t e) const {
        node->is_leaf = true;
        node->points.assign(points_storage.begin() + b, points_storage.begin() + e);
    }

    std::unique_ptr<KDNode<T>> buildIterative(Bounds<T> root_bounds) {
        struct Task { KDNode<T>* node; int depth; Bounds<T> bounds; size_t b, e; };
        std::vector<Task> stack;

        auto root = std::make_unique<KDNode<T>>();
        root->bounds    = root_bounds;
        root->begin_idx = 0;
        root->end_idx   = points_storage.size();
        root->is_leaf   = false;

        stack.reserve(256);
        stack.push_back({root.get(), 0, root_bounds, 0, points_storage.size()});

        while (!stack.empty()) {
            auto [node, depth, bounds, b, e] = stack.back();
            stack.pop_back();

            const size_t cnt = e - b;
            if (cnt == 0) {
                make_leaf(node, b, e);
                continue;
            }

            const bool degenerate = is_degenerate_bounds(bounds);
            const long long mass = range_mass(b, e);

            if (degenerate || mass <= (long long)min_points) {
                make_leaf(node, b, e);
                continue;
            }

            const int axis = depth % 2;
            node->split_dim = axis;

            // Work in sorted coordinate order for stable integer cuts.
            sort_range_by_axis(b, e, axis);

            const T min_c = first_coord(b, axis);
            const T max_c = last_coord(e, axis);

            if (min_c == max_c) {
                make_leaf(node, b, e);
                continue;
            }

            const T cut = choose_integer_cut_weighted(b, e, axis, mass);
            node->split_val = cut;

            const size_t mid = lower_bound_coord(b, e, axis, cut);

            if (mid == b || mid == e) {
                // If this happens, either the data are effectively constant on this axis
                // or numeric/pathological; safest is to stop.
                make_leaf(node, b, e);
                continue;
            }

            Bounds<T> L, R;
            carve_child_bounds(bounds, axis, cut, L, R);

            node->left  = std::make_unique<KDNode<T>>();
            node->left->bounds    = L;
            node->left->begin_idx = b;
            node->left->end_idx   = mid;

            node->right = std::make_unique<KDNode<T>>();
            node->right->bounds    = R;
            node->right->begin_idx = mid;
            node->right->end_idx   = e;

            stack.push_back({node->right.get(), depth + 1, R, mid, e});
            stack.push_back({node->left.get(),  depth + 1, L, b,   mid});
        }

        return root;
    }

    // double leaf_null_mass(const KDNode<T>& node) const {
    //     if constexpr (std::is_integral<T>::value) {
    //         double q = 0.0;

    //         for (const auto& pr : node.points) {
    //             const int x = pr.first.x;
    //             const int y = pr.first.y;

    //             const double px = copula->cdf_x(x) - copula->cdf_x(x - 1);
    //             const double py = copula->cdf_y(y) - copula->cdf_y(y - 1);

    //             q += px * py;
    //         }

    //         return std::max(q, 1e-300);
    //     } else {
    //         const double root_area_u = compute_root_area_u();
    //         const double a = get_bin_area(node);
    //         return std::max(a / root_area_u, 1e-300);
    //     }
    // }

    double leaf_rect_null_mass(const KDNode<T>& node) const {
        if constexpr (std::is_integral<T>::value) {
            const int lo_x = node.bounds.min_x - 1;
            const int lo_y = node.bounds.min_y - 1;

            double x_min = copula->cdf_x(lo_x);
            double y_min = copula->cdf_y(lo_y);

            double x_max = copula->cdf_x(node.bounds.max_x);
            double y_max = copula->cdf_y(node.bounds.max_y);

            auto clamp01 = [](double u) {
                if (u < 0.0) return 0.0;
                if (u > 1.0) return 1.0;
                return u;
            };

            x_min = clamp01(x_min);
            y_min = clamp01(y_min);
            x_max = clamp01(x_max);
            y_max = clamp01(y_max);

            return std::max((x_max - x_min) * (y_max - y_min), 1e-300);
        } else {
            const double root_area_u = compute_root_area_u();
            return std::max(get_bin_area(node) / root_area_u, 1e-300);
        }
    }

    // Can I make the underlying storage here an eigen vector, and then just
    // push it through the NB calculation? Or maybe even populate it with points
    // and the corresponding NB beforehand? -> Take the two Eigen vectors,
    // calculate the NB, and then pop the points with (x, y, nb_x, nb_y)

    double calculate_p_xy(int count, double bin_area) const {
        return static_cast<double>(count) /
               (static_cast<double>(total_count) * bin_area);
    }

    void traverse_and_compute(const KDNode<T>* node,
                              const double root_area_u,
                              double& mi,
                              double& area,
                              double& chi2) const {
        if (!node) return;

        if (node->is_leaf) {
            const int bin_count = node->total_counts();
            if (bin_count == 0) return;

            const double p = static_cast<double>(bin_count) / static_cast<double>(total_count);

            if (mode_ == MiMode::Raw) {
                const double log_p = std::log(p);
                mi += p * log_p;
                return;
            }

            const double q = leaf_rect_null_mass(*node);
            mi += p * (std::log(p) - std::log(q));

            const double expected_count = static_cast<double>(total_count) * q;
            if (expected_count > 0.0) {
                const double diff = static_cast<double>(bin_count) - expected_count;
                chi2 += diff * diff / expected_count;
            }

            return;
        }

        traverse_and_compute(node->left.get(),  root_area_u, mi, area, chi2);
        traverse_and_compute(node->right.get(), root_area_u, mi, area, chi2);
    }
};

template <typename T>
KDTree<T>::KDTree(const std::vector<Point<T>>& points, Copula<T>* copula,
                  int min_points_per_leaf, bool zi)
    : min_points(min_points_per_leaf), copula(copula), zi(zi) {
    total_count = points.size();

    std::vector<std::pair<Point<T>, int>> unique_points(points.size());

    for (int i = 0; i < points.size(); i++) {
        unique_points[i] = std::make_pair(points[i], 1);
    }

    // These are the boundaries of the unit square for assumed U[0, 1] if passed
    // anything other than ints

    // Check that this is true for other normal mi where the uniform transform isn't done?

    Bounds<T> bounds = {0.0, 1.0, 0.0, 1.0};

    // root = build(unique_points, 0, bounds, zi);
    points_storage = std::move(unique_points);
    root = buildIterative(bounds);

    #ifdef DEBUG
    this->dumpSplittingValuesToCSV("debug_splits.csv");
    #endif
}

// Set mode raw, avoid dangling copula being a problem
template <typename T>
KDTree<T>::KDTree(const std::vector<Point<T>>& points,
                  int min_points_per_leaf, bool zi)
    : min_points(min_points_per_leaf), copula(copula), zi(zi) {
    total_count = points.size();

    this->set_mode(MiMode::Raw);

    std::vector<std::pair<Point<T>, int>> unique_points(points.size());

    for (int i = 0; i < points.size(); i++) {
        unique_points[i] = std::make_pair(points[i], 1);
    }

    Bounds<T> bounds = {0.0, 1.0, 0.0, 1.0};

    points_storage = std::move(unique_points);
    root = buildIterative(bounds);

    #ifdef DEBUG
    this->dumpSplittingValuesToCSV("debug_splits.csv");
    #endif
}

inline Bounds<int> get_bounds(const std::vector<Point<int>>& points) {
    Bounds<int> b;
    b.min_x = std::numeric_limits<int>::max();
    b.max_x = std::numeric_limits<int>::min();
    b.min_y = std::numeric_limits<int>::max();
    b.max_y = std::numeric_limits<int>::min();

    for (const auto& p : points) {
        if (p.x < b.min_x) b.min_x = p.x;
        if (p.x > b.max_x) b.max_x = p.x;
        if (p.y < b.min_y) b.min_y = p.y;
        if (p.y > b.max_y) b.max_y = p.y;
    }

    return b;
}

template <>
KDTree<int>::KDTree(const std::vector<Point<int>>& points, Copula<int>* copula,
                    int min_points_per_leaf, bool zi)
    : min_points(min_points_per_leaf), copula(copula), zi(zi) {
    total_count = points.size();

    std::vector<std::pair<Point<int>, int>> unique_points =
        count_duplicates_unordered(points);

    // std::vector<std::pair<Point<int>, int>> unique_points(points.size());

    // for (int i = 0; i < points.size(); i++) {
    //     unique_points[i] = std::make_pair(points[i], 1);
    // }

    // These are the boundaries of the input data that then get mapped to [0, 1,
    // 0, 1] when transformed via the CDF

    Bounds<int> bounds = get_bounds(points);

    // Not worth it?
    // auto unique_points = count_duplicates_hist(points,
    //                                    bounds.min_x, bounds.max_x,
    //                                    bounds.min_y, bounds.max_y);

    // root = build(unique_points, 0, bounds, zi);

    points_storage = std::move(unique_points);
    root = buildIterative(bounds);

    #ifdef DEBUG
    this->dumpSplittingValuesToCSV("debug_splits.csv");
    #endif
}

template <>
KDTree<int>::KDTree(const std::vector<Point<int>>& points,
                    int min_points_per_leaf, bool zi)
    : min_points(min_points_per_leaf), zi(zi) {
    total_count = points.size();

    this->set_mode(MiMode::Raw);

    std::vector<std::pair<Point<int>, int>> unique_points =
        count_duplicates_unordered(points);

    Bounds<int> bounds = get_bounds(points);

    points_storage = std::move(unique_points);
    root = buildIterative(bounds);

    #ifdef DEBUG
    this->dumpSplittingValuesToCSV("debug_splits.csv");
    #endif
}

// In KDTree<T> (specialized for int):

template <>
template<class InputIt>
KDTree<int>::KDTree(InputIt first, InputIt last,
       Copula<int>* copula,
       int min_points_per_leaf,
       bool zi)
    : min_points(min_points_per_leaf),
      copula(copula),
      zi(zi)
{
    total_count = std::distance(first, last);

    // 1) compute integer bounds in one pass
    // Do this externally once?
    int min_x = std::numeric_limits<int>::max(), max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max(), max_y = std::numeric_limits<int>::min();
    for (auto it = first; it != last; ++it) {
        const auto& p = *it;
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    Bounds<int> bounds{min_x, max_x, min_y, max_y};

    // 2) dense 2D histogram duplicate‑count
    int nx = max_x - min_x + 1, ny = max_y - min_y + 1;
    std::vector<int> hist(nx * ny, 0), touched;
    touched.reserve(total_count);

    for (auto it = first; it != last; ++it) {
        const auto& p = *it;
        int ix = p.x - min_x, iy = p.y - min_y;
        int idx = ix * ny + iy;
        if (hist[idx]++ == 0) touched.push_back(idx);
    }

    // 3) collect unique (x,y,count)
    std::vector<std::pair<Point<int>,int>> unique_points;
    unique_points.reserve(touched.size());
    for (int idx : touched) {
        int ix = (idx / ny) + min_x;
        int iy = (idx % ny) + min_y;
        unique_points.emplace_back(Point<int>{ix, iy}, hist[idx]);
    }

    // 4) build tree
    // root = build(unique_points, 0, bounds, zi);
    points_storage = std::move(unique_points);
    root = buildIterative(bounds);

  #ifdef DEBUG
    dumpSplittingValuesToCSV("debug_splits.csv");
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
    const int lo_x = node.bounds.min_x - 1;
    const int lo_y = node.bounds.min_y - 1;

    double x_min = copula->cdf_x(lo_x);
    double y_min = copula->cdf_y(lo_y);

    double x_max = copula->cdf_x(node.bounds.max_x);
    double y_max = copula->cdf_y(node.bounds.max_y);

    auto clamp01 = [](double u) {
        if (u < 0.0) return 0.0;
        if (u > 1.0) return 1.0;
        return u;
    };

    x_min = clamp01(x_min); y_min = clamp01(y_min);
    x_max = clamp01(x_max); y_max = clamp01(y_max);

    const double width  = x_max - x_min;
    const double height = y_max - y_min;
    return width * height;
}

template<typename T>
template<class F>
void KDTree<T>::for_each_point(F&& emit) const {
    for_each_point_impl(root.get(), std::forward<F>(emit));
}

template<typename T>
template<class F>
void KDTree<T>::for_each_point_impl(const KDNode<int>* node, F&& emit) const {
    if (!node) return;

    if (node->is_leaf) {
        // node->points holds unique (x,y) with multiplicity in .second
        for (const auto& pr : node->points) {
            const int x = pr.first.x;
            const int y = pr.first.y;
            const int c = pr.second;
            if (c > 0) {
                // emit one discrete atom: (x, y, count)
                emit(x, y, c);
            }
        }
        return;
    }

    for_each_point_impl(node->left .get(), std::forward<F>(emit));
    for_each_point_impl(node->right.get(), std::forward<F>(emit));
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

template <typename T>
template <class InputIt>
std::pair<double, int> KDTree<T>::chi2_holdout(InputIt first, InputIt last, int n_test, double min_expected) const {
    if (!leaf_cache_ready) build_leaf_cache();
    const int L = static_cast<int>(leaf_nodes.size());
    if (L <= 1 || n_test <= 0) return {0.0, 0};

    std::vector<int> obs(L, 0);

    for (auto it = first; it != last; ++it) {
        const Point<T> p = *it;
        const int id = locate_leaf_id(p);
        if (id >= 0) obs[id] += 1;
    }

    double chi2 = 0.0;

    long long O_small = 0;
    double E_small = 0.0;
    int bins_kept = 0;

    for (int l = 0; l < L; ++l) {
        const double E = static_cast<double>(n_test) * leaf_prob[l];
        const double O = static_cast<double>(obs[l]);

        if (!(E > 0.0)) continue;

        if (E < min_expected) {
            O_small += obs[l];
            E_small += E;
        } else {
            const double d = O - E;
            chi2 += d * d / E;
            bins_kept += 1;
        }
    }

    if (E_small > 0.0) {
        const double d = static_cast<double>(O_small) - E_small;
        chi2 += d * d / E_small;
        bins_kept += 1;
    }

    const int df = (bins_kept > 0) ? (bins_kept - 1) : 0;
    return {chi2, df};
}
