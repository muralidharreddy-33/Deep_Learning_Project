#include "../include/relu.h"
#include "../include/file_utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

void ReLULayer::load_bin_data(const std::string& file_path, std::vector<float>& data) {
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

void ReLULayer::save_bin_data(const std::string& file_path, const std::vector<float>& data) {
    std::ofstream file(file_path, std::ios::binary); // Open file in binary mode
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    // Write the binary data to the file
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    file.close();
}

ReLULayer::ReLULayer(const std::string& input_path) {
    load_data(input_path, input);
}

void ReLULayer::execute(const std::string& output_path) {
    std::vector<float> output(input.size());

    // Apply ReLU activation function
    std::transform(input.begin(), input.end(), output.begin(), [](float x) { return std::max(0.0f, x); });

    // Save the output to the specified path
    save_data(output_path, output);
    std::cout << "ReLU layer executed and output saved to " << output_path << std::endl;
}