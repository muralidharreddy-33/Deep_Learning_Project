#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include<bits/stdc++.h>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_config.json>" << std::endl;
        return -1;
    }
    // Path to the model_config.json file from command-line argument
    std::string config_file_path = argv[1];

    // Open the JSON file
    std::ifstream config_file(config_file_path);
    if (!config_file.is_open()) {
        std::cerr << "Failed to open the model configuration file: " << config_file_path << std::endl;
        return -1;
    }

    // Parse the JSON file
    json model_config;
    config_file >> model_config;
    

    // Iterate through the layers
    for (const auto& layer : model_config["layers"]) {
        std::string layer_type = layer["type"];
        std::string layer_name = layer["name"];

        std::cout << "Processing layer: " << layer_name << " (" << layer_type << ")" << std::endl;

        if (layer_type == "Conv2D") {
            std::string kernel_file = layer["kernel"]["file"];
            std::vector<int> kernel_shape = layer["kernel"]["shape"];
            std::string bias_file = layer["bias"]["file"];
            std::vector<int> bias_shape = layer["bias"]["shape"];

        } else if (layer_type == "Dense") {
            std::string kernel_file = layer["kernel"]["file"];
            std::vector<int> kernel_shape = layer["kernel"]["shape"];
            std::string bias_file = layer["bias"]["file"];
            std::vector<int> bias_shape = layer["bias"]["shape"];
            std::string activation = layer["activation"];
            std::string output_file = layer["output_file"];
            std::string reference_file = layer["reference_file"];

        }
    }

    std::cout << "All layers processed successfully!" << std::endl;
    return 0;
}

// Function to load a vector from a .bin file
std::vector<float> load_binary_file(const std::string& file) {
    // Open the file in binary mode
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + file);
    }

    // Get the size of the file
    in.seekg(0, std::ios::end);
    size_t size = in.tellg();
    in.seekg(0, std::ios::beg);

    // Check if the file size is a multiple of sizeof(float)
    if (size % sizeof(float) != 0) {
        throw std::runtime_error("File size is not a multiple of sizeof(float): " + file);
    }

    // Create a vector to hold the data
    std::vector<float> data(size / sizeof(float));

    // Read the data from the file into the vector
    in.read(reinterpret_cast<char*>(data.data()), size);

    // Check if the read operation was successful
    if (!in) {
        throw std::runtime_error("Failed to read file: " + file);
    }

    return data;
}