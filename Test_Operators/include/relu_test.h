#ifndef RELU_TEST_H
#define RELU_TEST_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath> // For floating-point comparison

class ReLUTest {
public:
    ReLUTest(const std::string& input_path, const std::string& ref_path);
    bool run_test();

private:
    std::string input_path;
    std::string ref_path;

    void load_bin_data(const std::string& path, std::vector<float>& data);
    bool compare_output(const std::vector<float>& output, const std::vector<float>& ref_output);
};

#endif // RELU_TEST_H