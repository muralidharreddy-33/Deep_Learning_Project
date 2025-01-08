#ifndef FLATTEN_H
#define FLATTEN_H

#include <vector>

class FlattenLayer {
public:
    std::vector<float> forward(const std::vector<float>& input);
};

#endif // FLATTEN_H