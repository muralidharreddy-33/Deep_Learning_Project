#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../include/conv.h"
#include <chrono>
#include <iomanip>
#include <ctime>

using json = nlohmann::json;

// Function to get the current timestamp as a string
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
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