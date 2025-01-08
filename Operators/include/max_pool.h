#ifndef MAX_POOL_H
#define MAX_POOL_H

#include <vector>

class MaxPoolLayer {
public:
    std::vector<float> forward(const std::vector<float>& input, const std::vector<size_t>& input_shape);
};

#endif // MAX_POOL_H