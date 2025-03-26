#include <ruby.h>
#include <functional>
#include "trees.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

// Helper to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Implementation of RubyTreeDef methods
bool RubyTreeDef::is_leaf(VALUE obj) {
  // Check if the object is an MLX array or a Ruby non-enumerable
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    return true;
  }
  
  // For simplicity, we consider non-array/hash objects as leaves
  if (!rb_obj_is_kind_of(obj, rb_cArray) && !rb_obj_is_kind_of(obj, rb_cHash)) {
    return true;
  }
  
  return false;
}

std::pair<std::vector<VALUE>, std::vector<TreePath>> RubyTreeDef::flatten(VALUE obj) {
  std::vector<VALUE> leaves;
  std::vector<TreePath> paths;
  
  std::function<void(VALUE, TreePath)> traverse = [&](VALUE node, TreePath path) {
    if (is_leaf(node)) {
      leaves.push_back(node);
      paths.push_back(path);
      return;
    }
    
    if (rb_obj_is_kind_of(node, rb_cArray)) {
      long length = RARRAY_LEN(node);
      for (long i = 0; i < length; i++) {
        VALUE item = rb_ary_entry(node, i);
        TreePath new_path = path;
        new_path.push_back(i);
        traverse(item, new_path);
      }
    } else if (rb_obj_is_kind_of(node, rb_cHash)) {
      // For Hash, we need to sort keys for consistency
      VALUE keys = rb_funcall(node, rb_intern("keys"), 0);
      VALUE sorted_keys = rb_funcall(keys, rb_intern("sort"), 0);
      
      long length = RARRAY_LEN(sorted_keys);
      for (long i = 0; i < length; i++) {
        VALUE key = rb_ary_entry(sorted_keys, i);
        VALUE item = rb_hash_aref(node, key);
        TreePath new_path = path;
        new_path.push_back(i);
        traverse(item, new_path);
      }
    }
  };
  
  traverse(obj, {});
  return {leaves, paths};
}

VALUE RubyTreeDef::unflatten(const std::vector<VALUE>& leaves, const std::vector<TreePath>& paths) {
  if (leaves.empty()) {
    return Qnil;
  }
  
  if (leaves.size() == 1 && paths[0].empty()) {
    return leaves[0];
  }
  
  // First, determine the structure and create it
  std::function<VALUE(const TreePath&, size_t)> create_structure = [&](const TreePath& path, size_t depth) -> VALUE {
    if (depth >= path.size()) {
      return Qnil;
    }
    
    size_t max_index = 0;
    for (const auto& p : paths) {
      if (p.size() > depth && p[depth] > max_index) {
        max_index = p[depth];
      }
    }
    
    // For simplicity, we'll always use arrays
    VALUE result = rb_ary_new2(max_index + 1);
    for (long i = 0; i <= static_cast<long>(max_index); i++) {
      rb_ary_store(result, i, Qnil);
    }
    
    return result;
  };
  
  // Create the root structure
  VALUE root = create_structure(paths[0], 0);
  
  // Now fill in the values
  for (size_t i = 0; i < leaves.size(); i++) {
    const TreePath& path = paths[i];
    VALUE leaf = leaves[i];
    
    if (path.empty()) {
      return leaf;  // Single leaf case
    }
    
    VALUE current = root;
    for (size_t j = 0; j < path.size() - 1; j++) {
      size_t idx = path[j];
      VALUE next = rb_ary_entry(current, idx);
      
      if (NIL_P(next)) {
        next = create_structure(path, j + 1);
        rb_ary_store(current, idx, next);
      }
      
      current = next;
    }
    
    // Store the leaf
    rb_ary_store(current, path.back(), leaf);
  }
  
  return root;
}

// Tree module methods
static VALUE tree_flatten(VALUE self, VALUE obj) {
  auto [leaves, paths] = RubyTreeDef::flatten(obj);
  
  // Convert to Ruby format
  VALUE result = rb_ary_new2(2);
  
  // Create the leaves array
  VALUE rb_leaves = rb_ary_new2(leaves.size());
  for (size_t i = 0; i < leaves.size(); i++) {
    rb_ary_store(rb_leaves, i, leaves[i]);
  }
  
  // Create the paths array (array of arrays)
  VALUE rb_paths = rb_ary_new2(paths.size());
  for (size_t i = 0; i < paths.size(); i++) {
    const TreePath& path = paths[i];
    VALUE rb_path = rb_ary_new2(path.size());
    
    for (size_t j = 0; j < path.size(); j++) {
      rb_ary_store(rb_path, j, INT2NUM(path[j]));
    }
    
    rb_ary_store(rb_paths, i, rb_path);
  }
  
  rb_ary_store(result, 0, rb_leaves);
  rb_ary_store(result, 1, rb_paths);
  
  return result;
}

static VALUE tree_unflatten(VALUE self, VALUE leaves, VALUE paths) {
  Check_Type(leaves, T_ARRAY);
  Check_Type(paths, T_ARRAY);
  
  // Convert from Ruby format
  std::vector<VALUE> cpp_leaves;
  std::vector<TreePath> cpp_paths;
  
  for (long i = 0; i < RARRAY_LEN(leaves); i++) {
    cpp_leaves.push_back(rb_ary_entry(leaves, i));
  }
  
  for (long i = 0; i < RARRAY_LEN(paths); i++) {
    VALUE rb_path = rb_ary_entry(paths, i);
    Check_Type(rb_path, T_ARRAY);
    
    TreePath path;
    for (long j = 0; j < RARRAY_LEN(rb_path); j++) {
      VALUE index = rb_ary_entry(rb_path, j);
      path.push_back(NUM2INT(index));
    }
    
    cpp_paths.push_back(path);
  }
  
  return RubyTreeDef::unflatten(cpp_leaves, cpp_paths);
}

static VALUE tree_map(VALUE self, VALUE tree, VALUE func) {
  // Flatten the tree
  auto [leaves, paths] = RubyTreeDef::flatten(tree);
  
  // Apply the function to each leaf
  std::vector<VALUE> new_leaves;
  for (VALUE leaf : leaves) {
    VALUE result = rb_funcall(func, rb_intern("call"), 1, leaf);
    new_leaves.push_back(result);
  }
  
  // Unflatten with the same structure
  return RubyTreeDef::unflatten(new_leaves, paths);
}

// Initialize trees module
void init_trees(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "tree_flatten", RUBY_METHOD_FUNC(tree_flatten), 1);
  rb_define_module_function(module, "tree_unflatten", RUBY_METHOD_FUNC(tree_unflatten), 2);
  rb_define_module_function(module, "tree_map", RUBY_METHOD_FUNC(tree_map), 2);
} 