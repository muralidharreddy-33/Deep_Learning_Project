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
using json = nlohmann::json;

// Function to save a vector to a .bin file
void save_binary_file(const std::string& file, const std::vector<float>& data) {
    std::ofstream out(file, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file: " + file);
    }
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    // Print start timestamp
    std::cout << "[" << getCurrentTimestamp() << "] Program started." << std::endl;

    // Load the configuration file
    std::ifstream config_file(argv[1]);
    if (!config_file) {
        std::cerr << "[" << getCurrentTimestamp() << "] Error opening config file: " << argv[1] << std::endl;
        return 1;
    }

    json config;
    config_file >> config;

    // Execute layers based on the configuration
    for (const auto& layer : config["layers"]) {
        std::cout << "[" << getCurrentTimestamp() << "] Executing layer: " << layer["type"] << std::endl;

        if (layer["type"] == "conv") {
            // Ensure paths in the JSON file point to .bin files
            ConvLayer conv(
                layer["input_path"],   // Path to input .bin file
                layer["weights_path"], // Path to weights .bin file
                layer["bias_path"],    // Path to bias .bin file
                layer["attributes"]["input_height"],
                layer["attributes"]["input_width"],
                layer["attributes"]["input_channels"],
                layer["attributes"]["kernel_height"],
                layer["attributes"]["kernel_width"],
                layer["attributes"]["output_channels"],
                layer["attributes"]["stride"],
                layer["attributes"]["padding"]
            );
            conv.execute(layer["output_path"]); // Path to output .bin file
        } 
        else {
            std::cerr << "[" << getCurrentTimestamp() << "] Unknown layer type: " << layer["type"] << std::endl;
        }
    }

    // Print completion timestamp
    std::cout << "[" << getCurrentTimestamp() << "] Model execution completed!" << std::endl;
    return 0;
}