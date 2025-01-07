#ifndef RELU_H
#define RELU_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class ReLULayer {
public:
    ReLULayer(const std::string& input_path);
    void execute(const std::string& output_path);

private:
    std::vector<float> input;

    void load_bin_data(const std::string& path, std::vector<float>& data);
    void save_bin_data(const std::string& path, const std::vector<float>& data);
};

#endif // RELU_H