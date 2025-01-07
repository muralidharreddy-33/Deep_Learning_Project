#include "../include/relu_test.h"
#include "../include/file_utils.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

ReLUTest::ReLUTest(const std::string& input_path, const std::string& ref_path)
    : input_path(input_path), ref_path(ref_path) {}

void ReLUTest::load_bin_data(const std::string& file_path, std::vector<float>& data) {
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

bool ReLUTest::run_test() {
    std::vector<float> input, ref_output;

    // Load data from binary files
    load_data(input_path, input);
    load_data(ref_path, ref_output);

    // Apply ReLU activation function
    std::vector<float> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](float x) { return std::max(0.0f, x); });

    // Compare the output with the reference output
    return compare_output(output, ref_output);
}

bool ReLUTest::compare_output(const std::vector<float>& output, const std::vector<float>& ref_output) {
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

    std::cout << "ReLU test passed!" << std::endl;
    return true;
}