#pragma once

#include <ruby.h>
#include <functional>
#include <vector>
#include <utility>
#include "mlx/array.h"

// Forward declarations for required types
namespace mx = mlx::core;

// Function declarations for tree operations
void tree_visit(VALUE tree, std::function<void(VALUE)> visitor);
VALUE tree_visit_update(VALUE tree, std::function<VALUE(VALUE)> visitor);
VALUE tree_map(VALUE tree, std::function<VALUE(VALUE)> transform);

// Fill a tree with arrays
void tree_fill(VALUE tree, const std::vector<mx::array>& values);

// Replace arrays in a tree
void tree_replace(VALUE tree, 
                 const std::vector<mx::array>& src_arrays,
                 const std::vector<mx::array>& dst_arrays);

// Flatten a tree into a vector of arrays
std::vector<mx::array> tree_flatten(VALUE tree, bool strict = true);

// Unflatten arrays into a tree (default index is 0)
VALUE tree_unflatten(VALUE tree, const std::vector<mx::array>& values, int index = 0);

// Create a sentinel value for structure preservation
VALUE tree_sentinel_value();

// Flatten a tree and capture its structure
std::pair<std::vector<mx::array>, VALUE> tree_flatten_with_structure(VALUE tree, bool strict = true);

// Unflatten arrays using a structure
VALUE tree_unflatten_from_structure(VALUE structure, 
                                  const std::vector<mx::array>& values,
                                  int index = 0);

// Ruby wrapper functions - used in init_trees
VALUE rb_tree_flatten(VALUE self, VALUE tree);
VALUE rb_tree_unflatten(VALUE self, VALUE tree, VALUE values);
VALUE rb_tree_map(VALUE self, VALUE tree, VALUE func);
VALUE rb_tree_fill(VALUE self, VALUE tree, VALUE values);
VALUE rb_tree_replace(VALUE self, VALUE tree, VALUE src_values, VALUE dst_values);
VALUE rb_tree_flatten_arrays(int argc, VALUE* argv, VALUE self);
VALUE rb_tree_flatten_with_structure(int argc, VALUE* argv, VALUE self);
VALUE rb_tree_unflatten_from_structure(int argc, VALUE* argv, VALUE self);

// Initialize the trees module
void init_trees(VALUE module); 