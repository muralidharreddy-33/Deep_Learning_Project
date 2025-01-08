#include "../include/conv.h"
#include <fstream>
#include <stdexcept>

ConvLayer::ConvLayer(const std::string& kernel_file, const std::string& bias_file,
                     const std::vector<size_t>& kernel_shape, const std::vector<size_t>& bias_shape)
    : kernel_shape_(kernel_shape), bias_shape_(bias_shape) {
    size_t kernel_size = 1;
    for (size_t dim : kernel_shape) {
        kernel_size *= dim;
    }
    kernel_ = load_binary_file(kernel_file, kernel_size);

    size_t bias_size = 1;
    for (size_t dim : bias_shape) {
        bias_size *= dim;
    }
    bias_ = load_binary_file(bias_file, bias_size);
}

std::vector<float> ConvLayer::forward(const std::vector<float>& input, const std::vector<size_t>& input_shape) {
    size_t input_height = input_shape[0];
    size_t input_width = input_shape[1];
    size_t input_channels = input_shape[2];

    size_t kernel_height = kernel_shape_[0];
    size_t kernel_width = kernel_shape_[1];
    size_t kernel_channels = kernel_shape_[2];
    size_t num_filters = kernel_shape_[3];

    size_t output_height = input_height - kernel_height + 1;
    size_t output_width = input_width - kernel_width + 1;

    std::vector<float> output(output_height * output_width * num_filters, 0.0f);

    for (size_t f = 0; f < num_filters; ++f) {
        for (size_t i = 0; i < output_height; ++i) {
            for (size_t j = 0; j < output_width; ++j) {
                float sum = 0.0f;
                for (size_t c = 0; c < kernel_channels; ++c) {
                    for (size_t ki = 0; ki < kernel_height; ++ki) {
                        for (size_t kj = 0; kj < kernel_width; ++kj) {
                            size_t input_index = (i + ki) * input_width * input_channels + (j + kj) * input_channels + c;
                            size_t kernel_index = ki * kernel_width * kernel_channels * num_filters +
                                                 kj * kernel_channels * num_filters + c * num_filters + f;
                            sum += input[input_index] * kernel_[kernel_index];
                        }
                    }
                }
                sum += bias_[f];
                size_t output_index = i * output_width * num_filters + j * num_filters + f;
                output[output_index] = sum;
            }
        }
    }

    return output;
}

std::vector<float> ConvLayer::load_binary_file(const std::string& file, size_t size) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + file);
    }

    std::vector<float> data(size);
    in.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    if (!in) {
        throw std::runtime_error("Failed to read file: " + file);
    }

    return data;
}