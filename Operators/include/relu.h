#ifndef RELU_H
#define RELU_H

#include <vector>

class ReLU {
public:
    std::vector<float> forward(const std::vector<float>& input);
};

#endif // RELU_H