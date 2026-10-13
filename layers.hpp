#pragma once
// Queries used by graph optimization passes to recognise fusable layers
// without depending on the concrete layer classes in layers.cpp.
#include "conv.hpp"
#include "layer.hpp"

// Activation applied by a standalone activation layer (Swish/Sigmoid/ReLU), or None.
Activation unary_activation(const Layer& l);
// True for a two-input elementwise BinaryOp "add" (fusable as a residual).
bool is_binary_add(const Layer& l);
