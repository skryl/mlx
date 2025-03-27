#pragma once

#include <ruby.h>
#include "mlx/array.h"

namespace mx = mlx::core;

// Advanced indexing functions for array support
mx::array ruby_get_item(const mx::array& src, VALUE obj);
void ruby_set_item(mx::array& src, VALUE obj, VALUE update_val);
mx::array ruby_add_item(const mx::array& src, VALUE obj, VALUE update_val);
mx::array ruby_subtract_item(const mx::array& src, VALUE obj, VALUE update_val);
mx::array ruby_multiply_item(const mx::array& src, VALUE obj, VALUE update_val);
mx::array ruby_divide_item(const mx::array& src, VALUE obj, VALUE update_val);
mx::array ruby_maximum_item(const mx::array& src, VALUE obj, VALUE update_val);
mx::array ruby_minimum_item(const mx::array& src, VALUE obj, VALUE update_val);

// Helper functions for advanced indexing
std::tuple<std::vector<mx::array>, mx::array, std::vector<int>> 
compute_scatter_args(const mx::array& src, VALUE obj, VALUE update_val);

// Initialize the indexing module
void init_indexing(VALUE module); 