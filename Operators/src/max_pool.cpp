#include "../include/max_pool.h"
#include <limits> // Add this line

std::vector<float> MaxPoolLayer::forward(const std::vector<float>& input, const std::vector<size_t>& input_shape) {
    size_t input_height = input_shape[0];
    size_t input_width = input_shape[1];
    size_t input_channels = input_shape[2];

    size_t output_height = input_height / 2;
    size_t output_width = input_width / 2;

    std::vector<float> output(output_height * output_width * input_channels, 0.0f);

    for (size_t c = 0; c < input_channels; ++c) {
        for (size_t i = 0; i < output_height; ++i) {
            for (size_t j = 0; j < output_width; ++j) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t pi = 0; pi < 2; ++pi) {
                    for (size_t pj = 0; pj < 2; ++pj) {
                        size_t input_index = (i * 2 + pi) * input_width * input_channels + (j * 2 + pj) * input_channels + c;
                        max_val = std::max(max_val, input[input_index]);
                    }
                }
                size_t output_index = i * output_width * input_channels + j * input_channels + c;
                output[output_index] = max_val;
            }
        }
    }

    return output;
}