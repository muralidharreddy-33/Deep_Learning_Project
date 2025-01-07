#include "../include/conv_test.h"
#include "../include/file_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

ConvTest::ConvTest(const std::string& input_path, const std::string& weights_path, const std::string& bias_path, const std::string& ref_path)
    : input_path(input_path), weights_path(weights_path), bias_path(bias_path), ref_path(ref_path) {}

void ConvTest::load_bin_data(const std::string& file_path, std::vector<float>& data) {
    std::ifstream file(file_path, std::ios::binary); // Open file in binary mode
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    // Get the file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Resize the vector to hold the data
    data.resize(file_size / sizeof(float));

    // Read the binary data into the vector
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();
}

// Function to perform 2D convolution
std::vector<float> convolve2D(const std::vector<float>& input, const std::vector<float>& kernel,
                              int input_height, int input_width, int kernel_height, int kernel_width) {
    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;
    std::vector<float> output(output_height * output_width, 0.0f);

    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel_height; ++ki) {
                for (int kj = 0; kj < kernel_width; ++kj) {
                    sum += input[(i + ki) * input_width + (j + kj)] * kernel[ki * kernel_width + kj];
                }
            }
            output[i * output_width + j] = sum;
        }
    }

    return output;
}

bool ConvTest::run_test() {
    try {
        std::vector<float> input, weights, bias, ref_output;

        // Debugging: Print file paths
        std::cout << "Input file path: " << input_path << std::endl;
        std::cout << "Weights file path: " << weights_path << std::endl;
        std::cout << "Bias file path: " << bias_path << std::endl;
        std::cout << "Reference output file path: " << ref_path << std::endl;

        // Load data from binary files
        load_bin_data(input_path, input);
        load_bin_data(weights_path, weights);
        load_bin_data(bias_path, bias);
        load_bin_data(ref_path, ref_output);

        // Debugging: Print loaded data
        std::cout << "Input data size: " << input.size() << std::endl;
        std::cout << "Weights data size: " << weights.size() << std::endl;
        std::cout << "Bias data size: " << bias.size() << std::endl;
        std::cout << "Reference output data size: " << ref_output.size() << std::endl;

        // Define input and kernel dimensions
        int input_height = 4; // Example: Input height (adjust based on your data)
        int input_width = 4;  // Example: Input width (adjust based on your data)
        int kernel_height = 3; // Example: Kernel height (adjust based on your data)
        int kernel_width = 3;  // Example: Kernel width (adjust based on your data)

        // Debugging: Print dimensions
        std::cout << "Input dimensions: " << input_height << "x" << input_width << std::endl;
        std::cout << "Kernel dimensions: " << kernel_height << "x" << kernel_width << std::endl;

        // Perform 2D convolution
        std::vector<float> output = convolve2D(input, weights, input_height, input_width, kernel_height, kernel_width);
        
        // Debugging: Print output dimensions
        int output_height = input_height - kernel_height + 1;
        int output_width = input_width - kernel_width + 1;
        std::cout << "Output dimensions: " << output_height << "x" << output_width << std::endl;

        // Add bias to the output (if applicable)
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += bias[i % bias.size()];
        }

        // Debugging: Print output
        std::cout << "Computed output size: " << output.size() << std::endl;

        // Compare the output with the reference output
        std::cout<<"muralii"<<std::endl;
        std::cout<<output.size()<<ref_output.size()<<std::endl;
        return compare_output(output, ref_output);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

bool ConvTest::compare_output(const std::vector<float>& output, const std::vector<float>& ref_output) {
    if (output.size() != ref_output.size()) {
        std::cerr << "Error: Output and reference sizes do not match!" << std::endl;
        return false;
    }

    float tolerance = 1e-5;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::abs(output[i] - ref_output[i]) > tolerance) {
            std::cerr << "Error: Output mismatch at index " << i << "!" << std::endl;
            return false;
        }
    }

    std::cout << "Convolution test passed!" << std::endl;
    return true;
}