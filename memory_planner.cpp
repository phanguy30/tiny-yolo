#include "memory_planner.hpp"

#include <algorithm>
#include <numeric>

std::size_t plan_buffers(std::vector<PlanBuffer>& buffers, std::size_t alignment) {
    auto align = [alignment](std::size_t v) { return (v + alignment - 1) / alignment * alignment; };
    std::vector<std::size_t> order(buffers.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (buffers[a].size != buffers[b].size) return buffers[a].size > buffers[b].size;
        return buffers[a].first < buffers[b].first;
    });
    std::vector<std::size_t> placed;
    placed.reserve(buffers.size());
    std::vector<std::pair<std::size_t, std::size_t>> busy;  // [begin, end) of conflicting buffers
    std::size_t arena = 0;
    for (std::size_t idx : order) {
        PlanBuffer& b = buffers[idx];
        if (b.size == 0) { b.offset = 0; continue; }
        busy.clear();
        for (std::size_t p : placed) {
            const PlanBuffer& o = buffers[p];
            if (o.first <= b.last && b.first <= o.last) busy.emplace_back(o.offset, o.offset + align(o.size));
        }
        std::sort(busy.begin(), busy.end());
        std::size_t candidate = 0;
        for (const auto& r : busy) {
            if (r.first >= candidate + align(b.size)) break;  // fits in the gap before r
            candidate = std::max(candidate, r.second);
        }
        b.offset = candidate;
        arena = std::max(arena, candidate + align(b.size));
        placed.push_back(idx);
    }
    return arena;
}
