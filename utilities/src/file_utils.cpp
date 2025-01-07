#include "../include/file_utils.h"
#include <fstream>
#include <iostream>
#include <vector>

void load_data(const std::string& file_path, std::vector<float>& data) {
    std::ifstream infile(file_path, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return;
    }

    float value;
    while (infile.read(reinterpret_cast<char*>(&value), sizeof(float))) {
        data.push_back(value);
    }
    infile.close();
}

void save_data(const std::string& file_path, const std::vector<float>& data) {
    std::ofstream outfile(file_path, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return;
    }

    for (const float& value : data) {
        outfile.write(reinterpret_cast<const char*>(&value), sizeof(float));
    }
    outfile.close();
}
