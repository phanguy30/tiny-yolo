#pragma once
// Static memory planner: assigns every buffer an offset in one arena so that
// buffers whose lifetimes overlap never overlap in memory.  Uses the
// "greedy by size" heuristic (largest buffers first, lowest fitting offset).
#include <cstddef>
#include <vector>

struct PlanBuffer {
    std::size_t size = 0;  // bytes
    int first = 0, last = 0;  // inclusive op-index lifetime
    std::size_t offset = 0;   // result
};

// Returns the arena size in bytes.
std::size_t plan_buffers(std::vector<PlanBuffer>& buffers, std::size_t alignment = 64);
