#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <vector>

class Softmax {
public:
    std::vector<float> forward(const std::vector<float>& input);
};

#endif // SOFTMAX_H