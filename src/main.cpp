#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "conv.h"
#include "max_pool.h"
#include "flatten.h"
#include "dense.h"
#include "relu.h"
#include "softmax.h"
#include <random>

using json = nlohmann::json;

// Function to save a vector to a .bin file
void save_binary_file(const std::string& file, const std::vector<float>& data) {
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file: " + file);
    }
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}
// Function to load a vector from a .bin file
std::vector<float> load_binary_file(const std::string& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + file);
    }
    in.seekg(0, std::ios::end);
    size_t size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    in.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}
bool compare_outputs(const std::vector<float>& output, const std::vector<float>& reference, float tolerance = 0.001) {
    if (output.size() != reference.size()) {
        std::cerr << "Error: Output and reference sizes do not match!" << std::endl;
        return false;
    }
    std::cout<<output.size()<<" "<<reference.size()<<" "<<std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::abs(output[i] - reference[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
int main(int argc, char* argv[]) {
    // Check if the configuration file path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file_path>" << std::endl;
        return 1;
    }

    // Load the JSON configuration file
    std::string config_file_path = argv[1];
    std::ifstream config_file(config_file_path);
    if (!config_file) {
        std::cerr << "Error: Could not open " << config_file_path << std::endl;
        return 1;
    }

    json config;
    config_file >> config;

    // Example input (replace with actual input data)
    std::vector<float> input(32 * 32 * 3,1.0f); // Example input (32x32 image with 3 channels)
    std::vector<size_t> input_shape = {32, 32, 3};
    // Iterate over layers
    for (const auto& layer_config : config["layers"]) {
        std::string layer_type = layer_config["type"];
        std::string layer_name = layer_config["name"];
        std::cout << "Processing layer: " << layer_name << " (" << layer_type << ")" << std::endl;

        if (layer_type == "Conv2D") {
            // Extract kernel and bias shapes from JSON
            std::vector<size_t> kernel_shape = layer_config["kernel"]["shape"];
            std::vector<size_t> bias_shape = layer_config["bias"]["shape"];

            ConvLayer conv_layer(layer_config["kernel"]["file"], layer_config["bias"]["file"],
                                kernel_shape, bias_shape);
            input = conv_layer.forward(input, input_shape);

            ReLU relu;
            input = relu.forward(input);

            // Calculate output shape
            size_t output_height = input_shape[0] - kernel_shape[0] + 1;
            size_t output_width = input_shape[1] - kernel_shape[1] + 1;
            size_t num_filters = kernel_shape[3];
            input_shape = {output_height, output_width, num_filters};
            std::vector<float> reference_output = load_binary_file(layer_config["reference_file"]);
            // Compare outputs
            bool pass = compare_outputs(input, reference_output);
            std::cout << "Layer " << layer_name << " (" << layer_type << "): " << (pass ? "PASS" : "FAIL") << std::endl;

            save_binary_file(layer_config["output_file"], input);
        } else if (layer_type == "MaxPooling2D") {
            MaxPoolLayer max_pool;
            input = max_pool.forward(input, input_shape);

            // Calculate output shape
            size_t output_height = input_shape[0] / 2;
            size_t output_width = input_shape[1] / 2;
            size_t num_channels = input_shape[2];
            input_shape = {output_height, output_width, num_channels};
            // Load reference output
            std::vector<float> reference_output = load_binary_file(layer_config["reference_file"]);

            // Compare outputs
            bool pass = compare_outputs(input, reference_output);
            std::cout << "Layer " << layer_name << " (" << layer_type << "): " << (pass ? "PASS" : "FAIL") << std::endl;

            // Save output
            save_binary_file(layer_config["output_file"], input);
        } else if (layer_type == "Flatten") {
            FlattenLayer flatten;
            input = flatten.forward(input);

            input_shape = {input.size()};

            // Load reference output
            std::vector<float> reference_output = load_binary_file(layer_config["reference_file"]);

            // Compare outputs
            bool pass = compare_outputs(input, reference_output);
            std::cout << "Layer " << layer_name << " (" << layer_type << "): " << (pass ? "PASS" : "FAIL") << std::endl;

            // Save output
            save_binary_file(layer_config["output_file"], input);
        } else if (layer_type == "Dense") {
            // Extract kernel and bias shapes from JSON
            std::vector<size_t> kernel_shape = layer_config["kernel"]["shape"];
            std::vector<size_t> bias_shape = layer_config["bias"]["shape"];

            DenseLayer dense_layer(layer_config["kernel"]["file"], layer_config["bias"]["file"],
                                  kernel_shape, bias_shape);
            input = dense_layer.forward(input);

            if (layer_config["activation"] == "relu") {
                ReLU relu;
                input = relu.forward(input);
            } else if (layer_config["activation"] == "softmax") {
                Softmax softmax;
                input = softmax.forward(input);
            }

            input_shape = {input.size()};
             // Load reference output
            std::vector<float> reference_output = load_binary_file(layer_config["reference_file"]);

            // Compare outputs
            bool pass = compare_outputs(input, reference_output);
            std::cout << "Layer " << layer_name << " (" << layer_type << "): " << (pass ? "PASS" : "FAIL") << std::endl;
            // Save output
            save_binary_file(layer_config["output_file"], input);
        }
    }

    std::cout << "All layers processed successfully!" << std::endl;
    return 0;
}