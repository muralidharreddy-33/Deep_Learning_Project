#include "../include/dense.h"
#include <fstream>
#include <stdexcept>

DenseLayer::DenseLayer(const std::string& kernel_file, const std::string& bias_file,
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

std::vector<float> DenseLayer::forward(const std::vector<float>& input) {
    size_t input_size = kernel_shape_[0];
    size_t output_size = kernel_shape_[1];

    std::vector<float> output(output_size, 0.0f);

    for (size_t i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < input_size; ++j) {
            sum += input[j] * kernel_[j * output_size + i];
        }
        sum += bias_[i];
        output[i] = sum;
    }

    return output;
}

std::vector<float> DenseLayer::load_binary_file(const std::string& file, size_t size) {
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