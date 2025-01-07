#include "../include/conv_test.h"
#include "../include/relu_test.h"
#include <iostream>

int main() {
    try {
        // Test Convolution Layer
        ConvTest conv_test(
            "C:/Users/mcw/Downloads/PROJECT/data/input/conv2d_6_filters.bin",  // Input
            "C:/Users/mcw/Downloads/PROJECT/data/input/conv2d_6_filters.bin",  // Weights
            "C:/Users/mcw/Downloads/PROJECT/data/input/conv2d_6_biases.bin",   // Bias
            "C:/Users/mcw/Downloads/PROJECT/data/output/conv2d_6_output.bin"   // Reference output
        );
        if (!conv_test.run_test()) {
            std::cerr << "Convolution test failed!" << std::endl;
            return 1;
        }

        // Test ReLU Layer
        ReLUTest relu_test(
            "C:/Users/mcw/Downloads/PROJECT/data/output/conv2d_6_output.bin",   // Input (use output of conv2d_6 as input for ReLU)
            "C:/Users/mcw/Downloads/PROJECT/data/output/conv2d_6_output.bin"    // Reference output (same as input for ReLU)
        );
        if (!relu_test.run_test()) {
            std::cerr << "ReLU test failed!" << std::endl;
            return 1;
        }

        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}