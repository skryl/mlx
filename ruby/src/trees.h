#pragma once

#include <ruby.h>
#include <functional>
#include <vector>
#include "mlx/array.h"

// Tree path for specifying paths in a nested structure
using TreePath = std::vector<size_t>;

// Used to customize tree operations
struct RubyTreeDef {
  // Check if an object is a leaf node in the tree
  static bool is_leaf(VALUE obj);
  
  // Flatten a nested structure into a 1D vector of leaves and corresponding paths
  static std::pair<std::vector<VALUE>, std::vector<TreePath>> flatten(VALUE obj);
  
  // Rebuild a tree from flattened leaves and paths
  static VALUE unflatten(const std::vector<VALUE>& leaves, const std::vector<TreePath>& paths);
};

// Initialize the trees module
void init_trees(VALUE module); 