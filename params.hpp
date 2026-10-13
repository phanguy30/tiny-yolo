#pragma once
// Parsed ncnn layer parameters ("key=value" tokens). Parsed once at load time.
#include <string>
#include <unordered_map>
#include <vector>

class ParamDict {
public:
    void parse_token(const std::string& token);
    bool has(int key) const { return scalars_.count(key) || arrays_.count(key); }
    int get_int(int key, int fallback) const;
    float get_float(int key, float fallback) const;
    std::vector<float> get_array(int key) const;

private:
    std::unordered_map<int, double> scalars_;
    std::unordered_map<int, std::vector<float>> arrays_;
};
