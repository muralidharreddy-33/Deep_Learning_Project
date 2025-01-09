#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

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
            std::string activation = layer["activation"];
            std::string output_file = layer["output_file"];
            std::string reference_file = layer["reference_file"];

            std::cout << "  Kernel File: " << kernel_file << std::endl;
            std::cout << "  Kernel Shape: ";
            for (int dim : kernel_shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "  Bias File: " << bias_file << std::endl;
            std::cout << "  Bias Shape: ";
            for (int dim : bias_shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "  Activation: " << activation << std::endl;
            std::cout << "  Output File: " << output_file << std::endl;
            std::cout << "  Reference File: " << reference_file << std::endl;
        } else if (layer_type == "MaxPooling2D") {
            std::string output_file = layer["output_file"];
            std::string reference_file = layer["reference_file"];

            std::cout << "  Output File: " << output_file << std::endl;
            std::cout << "  Reference File: " << reference_file << std::endl;
        } else if (layer_type == "Flatten") {
            std::string output_file = layer["output_file"];
            std::string reference_file = layer["reference_file"];

            std::cout << "  Output File: " << output_file << std::endl;
            std::cout << "  Reference File: " << reference_file << std::endl;
        } else if (layer_type == "Dense") {
            std::string kernel_file = layer["kernel"]["file"];
            std::vector<int> kernel_shape = layer["kernel"]["shape"];
            std::string bias_file = layer["bias"]["file"];
            std::vector<int> bias_shape = layer["bias"]["shape"];
            std::string activation = layer["activation"];
            std::string output_file = layer["output_file"];
            std::string reference_file = layer["reference_file"];

            std::cout << "  Kernel File: " << kernel_file << std::endl;
            std::cout << "  Kernel Shape: ";
            for (int dim : kernel_shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "  Bias File: " << bias_file << std::endl;
            std::cout << "  Bias Shape: ";
            for (int dim : bias_shape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            std::cout << "  Activation: " << activation << std::endl;
            std::cout << "  Output File: " << output_file << std::endl;
            std::cout << "  Reference File: " << reference_file << std::endl;
        }
    }

    std::cout << "All layers processed successfully!" << std::endl;
    return 0;
}