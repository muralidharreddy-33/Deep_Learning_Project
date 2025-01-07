#ifndef CONV_TEST_H
#define CONV_TEST_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath> // For floating-point comparison

class ConvTest {
public:
    ConvTest(const std::string& input_path, const std::string& weights_path, const std::string& bias_path, const std::string& ref_path);
    bool run_test();

private:
    std::string input_path;
    std::string weights_path;
    std::string bias_path;
    std::string ref_path;

    void load_bin_data(const std::string& path, std::vector<float>& data);
    bool compare_output(const std::vector<float>& output, const std::vector<float>& ref_output);
};

#endif // CONV_TEST_H