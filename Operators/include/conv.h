#ifndef CONV_H
#define CONV_H

#include <vector>
#include <string>

class ConvLayer {
public:
    ConvLayer(const std::string& kernel_file, const std::string& bias_file,
              const std::vector<size_t>& kernel_shape, const std::vector<size_t>& bias_shape);
    std::vector<float> forward(const std::vector<float>& input, const std::vector<size_t>& input_shape);

private:
    std::vector<float> kernel_;
    std::vector<float> bias_;
    std::vector<size_t> kernel_shape_;
    std::vector<size_t> bias_shape_;

    std::vector<float> load_binary_file(const std::string& file, size_t size);
};

#endif // CONV_H