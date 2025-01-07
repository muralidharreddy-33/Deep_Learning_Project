#include "../include/conv.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

void ConvLayer::load_bin_data(const std::string& file_path, std::vector<float>& data) {
    std::ifstream file(file_path, std::ios::binary);
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

    // Debugging: Print file size and first few values
    std::cout << "Loaded file: " << file_path << ", size: " << file_size << " bytes, values: ";
    for (size_t i = 0; i < std::min(data.size(), size_t(5)); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void ConvLayer::save_bin_data(const std::string& file_path, const std::vector<float>& data) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    // Write the binary data to the file
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    file.close();
}

ConvLayer::ConvLayer(const std::string& input_path, const std::string& weights_path, const std::string& bias_path, 
                     int input_height, int input_width, int input_channels, 
                     int kernel_height, int kernel_width, int output_channels, 
                     int stride, const std::string& padding)
    : input_height(input_height), input_width(input_width), input_channels(input_channels),
      kernel_height(kernel_height), kernel_width(kernel_width), output_channels(output_channels),
      stride(stride), padding(padding) {
    load_bin_data(input_path, input);
    load_bin_data(weights_path, weights);
    load_bin_data(bias_path, bias);
}

void ConvLayer::execute(const std::string& output_path) {
    // Calculate output dimensions based on padding
    int output_height, output_width;
    if (padding == "same") {
        output_height = input_height;
        output_width = input_width;
    } else if (padding == "valid") {
        output_height = (input_height - kernel_height) / stride + 1;
        output_width = (input_width - kernel_width) / stride + 1;
    } else {
        std::cerr << "Invalid padding type: " << padding << std::endl;
        return;
    }

    // Debugging: Print dimensions
    std::cout << "Input dimensions: " << input_height << "x" << input_width << "x" << input_channels << std::endl;
    std::cout << "Kernel dimensions: " << kernel_height << "x" << kernel_width << "x" << input_channels << "x" << output_channels << std::endl;
    std::cout << "Output dimensions: " << output_height << "x" << output_width << "x" << output_channels << std::endl;

    // Initialize output tensor
    std::vector<float> output(output_height * output_width * output_channels, 0.0f);

    // Perform convolution
    for (int oc = 0; oc < output_channels; ++oc) { // Loop over output channels
        for (int oh = 0; oh < output_height; ++oh) { // Loop over output height
            for (int ow = 0; ow < output_width; ++ow) { // Loop over output width
                float sum = 0.0f;

                for (int kh = 0; kh < kernel_height; ++kh) { // Loop over kernel height
                    for (int kw = 0; kw < kernel_width; ++kw) { // Loop over kernel width
                        for (int ic = 0; ic < input_channels; ++ic) { // Loop over input channels
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;

                            // Handle padding
                            if (padding == "same") {
                                ih -= (kernel_height - 1) / 2;
                                iw -= (kernel_width - 1) / 2;
                            }

                            // Check if the input indices are within bounds
                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                int input_index = (ih * input_width + iw) * input_channels + ic;
                                int kernel_index = ((kh * kernel_width + kw) * input_channels + ic) * output_channels + oc;
                                sum += input[input_index] * weights[kernel_index];
                            }
                        }
                    }
                }

                // Add bias
                sum += bias[oc];

                // Store the result in the output tensor
                int output_index = (oh * output_width + ow) * output_channels + oc;
                output[output_index] = sum;
            }
        }
    }

    // Save the output to the specified path
    save_bin_data(output_path, output);
    std::cout << "Convolution layer executed and output saved to " << output_path << std::endl;
}