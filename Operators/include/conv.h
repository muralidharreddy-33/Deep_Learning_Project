#ifndef CONV_H
#define CONV_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class ConvLayer {
public:
    ConvLayer(const std::string& input_path, const std::string& weights_path, const std::string& bias_path, 
              int input_height, int input_width, int input_channels, 
              int kernel_height, int kernel_width, int output_channels, 
              int stride, const std::string& padding);
    void execute(const std::string& output_path);

private:
    std::vector<float> input;
    std::vector<float> weights;
    std::vector<float> bias;
    int input_height, input_width, input_channels;
    int kernel_height, kernel_width, output_channels;
    int stride;
    std::string padding;

    void load_bin_data(const std::string& path, std::vector<float>& data);
    void save_bin_data(const std::string& path, const std::vector<float>& data);
};

#endif // CONV_H