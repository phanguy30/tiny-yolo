#include "params.hpp"

#include <stdexcept>

void ParamDict::parse_token(const std::string& token) {
    const auto pos = token.find('=');
    if (pos == std::string::npos) throw std::runtime_error("bad param token: " + token);
    const int raw_key = std::stoi(token.substr(0, pos));
    const std::string value = token.substr(pos + 1);
    if (raw_key <= -23300) {  // array: "-233xx=count,v0,v1,..."
        std::vector<float> values;
        std::size_t start = 0;
        bool first = true;
        while (start <= value.size()) {
            std::size_t end = value.find(',', start);
            if (end == std::string::npos) end = value.size();
            const std::string item = value.substr(start, end - start);
            if (!first && !item.empty()) values.push_back(std::stof(item));
            first = false;
            start = end + 1;
        }
        arrays_[-raw_key - 23300] = std::move(values);
    } else {
        scalars_[raw_key] = std::stod(value);
    }
}

int ParamDict::get_int(int key, int fallback) const {
    auto it = scalars_.find(key);
    return it == scalars_.end() ? fallback : static_cast<int>(it->second);
}

float ParamDict::get_float(int key, float fallback) const {
    auto it = scalars_.find(key);
    return it == scalars_.end() ? fallback : static_cast<float>(it->second);
}

std::vector<float> ParamDict::get_array(int key) const {
    auto it = arrays_.find(key);
    return it == arrays_.end() ? std::vector<float>{} : it->second;
}
