#include "../include/softmax.h"
#include <cmath>

std::vector<float> Softmax::forward(const std::vector<float>& input) {
    std::vector<float> output = input;
    float sum = 0.0f;
    for (float val : output) {
        sum += std::exp(val);
    }
    for (float& val : output) {
        val = std::exp(val) / sum;
    }
    return output;
}