#include "../include/relu.h"

std::vector<float> ReLU::forward(const std::vector<float>& input) {
    std::vector<float> output = input;
    for (float& val : output) {
        val = std::max(0.0f, val);
    }
    return output;
}