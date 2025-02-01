#pragma once

template <typename T>
struct Point {
    T x;
    T y;

    bool operator<(const Point& other) const {
        return (x < other.x) || ((x == other.x) && (y < other.y));
    }
};
