#include "../include/flatten.h"

std::vector<float> FlattenLayer::forward(const std::vector<float>& input) {
    return input; // Flattening is just reshaping, so no change in data
}