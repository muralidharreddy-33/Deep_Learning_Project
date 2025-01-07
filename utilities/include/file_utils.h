#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>
#include <vector>

void load_data(const std::string& file_path, std::vector<float>& data);
void save_data(const std::string& file_path, const std::vector<float>& data);

#endif // FILE_UTILS_H
