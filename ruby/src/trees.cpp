#include <ruby.h>
#include <functional>
#include <unordered_map>
#include "trees.h"
#include "utils.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

// Check if an object is a leaf node in the tree
static bool is_leaf(VALUE obj) {
  // Check if the object is an MLX array
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    return true;
  }
  
  // For non-containers (not array or hash), treat as leaf
  if (!rb_obj_is_kind_of(obj, rb_cArray) && !rb_obj_is_kind_of(obj, rb_cHash)) {
    return true;
  }
  
  return false;
}

// Visit each node in a tree
void tree_visit(VALUE tree, std::function<void(VALUE)> visitor) {
  std::function<void(VALUE)> recurse = [&](VALUE node) {
    if (is_leaf(node)) {
      visitor(node);
      return;
    }
    
    if (rb_obj_is_kind_of(node, rb_cArray)) {
      long length = RARRAY_LEN(node);
      for (long i = 0; i < length; i++) {
        VALUE item = rb_ary_entry(node, i);
        recurse(item);
      }
    } else if (rb_obj_is_kind_of(node, rb_cHash)) {
      VALUE keys = rb_funcall(node, rb_intern("keys"), 0);
      long length = RARRAY_LEN(keys);
      for (long i = 0; i < length; i++) {
        VALUE key = rb_ary_entry(keys, i);
        VALUE item = rb_hash_aref(node, key);
        recurse(item);
      }
    }
  };
  
  recurse(tree);
}

// Visit and update each node in a tree
VALUE tree_visit_update(VALUE tree, std::function<VALUE(VALUE)> visitor) {
  std::function<VALUE(VALUE)> recurse = [&](VALUE node) {
    if (is_leaf(node)) {
      return visitor(node);
    }
    
    if (rb_obj_is_kind_of(node, rb_cArray)) {
      long length = RARRAY_LEN(node);
      VALUE result = rb_ary_new2(length);
      
      for (long i = 0; i < length; i++) {
        VALUE item = rb_ary_entry(node, i);
        VALUE updated = recurse(item);
        rb_ary_store(result, i, updated);
      }
      
      return result;
    } else if (rb_obj_is_kind_of(node, rb_cHash)) {
      VALUE result = rb_hash_new();
      VALUE keys = rb_funcall(node, rb_intern("keys"), 0);
      
      long length = RARRAY_LEN(keys);
      for (long i = 0; i < length; i++) {
        VALUE key = rb_ary_entry(keys, i);
        VALUE item = rb_hash_aref(node, key);
        VALUE updated = recurse(item);
        rb_hash_aset(result, key, updated);
      }
      
      return result;
    }
    
    return node;  // Fallback
  };
  
  return recurse(tree);
}

// Map a function over each leaf in a tree
VALUE tree_map(VALUE tree, std::function<VALUE(VALUE)> transform) {
  return tree_visit_update(tree, transform);
}

// Fill a tree with arrays
void tree_fill(VALUE tree, const std::vector<mx::array>& values) {
  size_t index = 0;
  tree_visit_update(tree, [&](VALUE node) -> VALUE {
    if (rb_obj_is_kind_of(node, rb_path2class("MLX::Core::Array"))) {
      if (index < values.size()) {
        return wrap_array(values[index++]);
      }
    }
    return node;
  });
}

// Replace arrays in a tree based on source/destination mapping
void tree_replace(VALUE tree, 
                 const std::vector<mx::array>& src_arrays,
                 const std::vector<mx::array>& dst_arrays) {
  if (src_arrays.size() != dst_arrays.size()) {
    rb_raise(rb_eArgError, "Source and destination arrays must have the same length");
  }
  
  std::unordered_map<uintptr_t, mx::array> src_to_dst;
  for (size_t i = 0; i < src_arrays.size(); ++i) {
    src_to_dst.insert({src_arrays[i].id(), dst_arrays[i]});
  }
  
  tree_visit_update(tree, [&](VALUE node) -> VALUE {
    if (rb_obj_is_kind_of(node, rb_path2class("MLX::Core::Array"))) {
      mx::array& arr = get_array(node);
      uintptr_t id = arr.id();
      
      auto it = src_to_dst.find(id);
      if (it != src_to_dst.end()) {
        return wrap_array(it->second);
      }
    }
    return node;
  });
}

// Flatten a tree into a vector of arrays
std::vector<mx::array> tree_flatten(VALUE tree, bool strict) {
  std::vector<mx::array> flat_arrays;
  
  tree_visit(tree, [&](VALUE obj) {
    if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
      flat_arrays.push_back(get_array(obj));
    } else if (strict && is_leaf(obj)) {
      rb_raise(rb_eArgError, "Tree contains non-MLX::Core::Array leaf values");
    }
  });
  
  return flat_arrays;
}

// Unflatten arrays into a tree
VALUE tree_unflatten(VALUE tree, const std::vector<mx::array>& values, int index) {
  if (index >= static_cast<int>(values.size()) && !values.empty()) {
    rb_raise(rb_eArgError, "Index out of bounds");
  }
  
  int current_index = index;
  return tree_visit_update(tree, [&](VALUE obj) -> VALUE {
    if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
      if (current_index < static_cast<int>(values.size())) {
        return wrap_array(values[current_index++]);
      }
    }
    return obj;
  });
}

// Create a sentinel value for structure preservation
VALUE tree_sentinel_value() {
  static VALUE sentinel = Qnil;
  
  if (NIL_P(sentinel)) {
    sentinel = rb_data_object_wrap(rb_cObject, nullptr, nullptr, nullptr);
  }
  
  return sentinel;
}

// Flatten a tree and capture its structure
std::pair<std::vector<mx::array>, VALUE> tree_flatten_with_structure(VALUE tree, bool strict) {
  std::vector<mx::array> flat_arrays;
  VALUE sentinel = tree_sentinel_value();
  
  VALUE structure = tree_visit_update(tree, [&](VALUE obj) -> VALUE {
    if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
      flat_arrays.push_back(get_array(obj));
      return sentinel;
    } else if (strict && !is_leaf(obj)) {
      rb_raise(rb_eArgError, "Tree contains non-MLX::Core::Array leaf values");
    }
    return obj;
  });
  
  return {flat_arrays, structure};
}

// Unflatten arrays using a structure
VALUE tree_unflatten_from_structure(VALUE structure, 
                                   const std::vector<mx::array>& values,
                                   int index) {
  if (index >= static_cast<int>(values.size()) && !values.empty()) {
    rb_raise(rb_eArgError, "Index out of bounds");
  }
  
  VALUE sentinel = tree_sentinel_value();
  int current_index = index;
  
  return tree_visit_update(structure, [&](VALUE obj) -> VALUE {
    if (obj == sentinel) {
      if (current_index < static_cast<int>(values.size())) {
        return wrap_array(values[current_index++]);
      } else {
        rb_raise(rb_eArgError, "Not enough arrays to unflatten");
      }
    }
    return obj;
  });
}

// Ruby wrapper functions for the C++ API

// Tree flatten function for Ruby
VALUE rb_tree_flatten(VALUE self, VALUE tree) {
  auto flat_arrays = tree_flatten(tree, true);
  
  VALUE result = rb_ary_new2(flat_arrays.size());
  for (size_t i = 0; i < flat_arrays.size(); i++) {
    rb_ary_store(result, i, wrap_array(flat_arrays[i]));
  }
  
  return result;
}

// Tree unflatten function for Ruby
VALUE rb_tree_unflatten(VALUE self, VALUE tree, VALUE values) {
  Check_Type(values, T_ARRAY);
  
  std::vector<mx::array> cpp_values;
  for (long i = 0; i < RARRAY_LEN(values); i++) {
    VALUE v = rb_ary_entry(values, i);
    if (!rb_obj_is_kind_of(v, rb_path2class("MLX::Core::Array"))) {
      rb_raise(rb_eTypeError, "Expected all elements to be MLX::Core::Array objects");
    }
    cpp_values.push_back(get_array(v));
  }
  
  return tree_unflatten(tree, cpp_values, 0);
}

// Tree map function for Ruby
VALUE rb_tree_map(VALUE self, VALUE tree, VALUE func) {
  if (!rb_respond_to(func, rb_intern("call"))) {
    rb_raise(rb_eTypeError, "Expected callable object");
  }
  
  return tree_visit_update(tree, [&](VALUE obj) -> VALUE {
    return rb_funcall(func, rb_intern("call"), 1, obj);
  });
}

// Tree fill function for Ruby
VALUE rb_tree_fill(VALUE self, VALUE tree, VALUE values) {
  Check_Type(values, T_ARRAY);
  
  std::vector<mx::array> cpp_values;
  for (long i = 0; i < RARRAY_LEN(values); i++) {
    VALUE v = rb_ary_entry(values, i);
    if (!rb_obj_is_kind_of(v, rb_path2class("MLX::Core::Array"))) {
      rb_raise(rb_eTypeError, "Expected all elements to be MLX::Core::Array objects");
    }
    cpp_values.push_back(get_array(v));
  }
  
  tree_fill(tree, cpp_values);
  return tree;
}

// Tree replace function for Ruby
VALUE rb_tree_replace(VALUE self, VALUE tree, VALUE src_values, VALUE dst_values) {
  Check_Type(src_values, T_ARRAY);
  Check_Type(dst_values, T_ARRAY);
  
  if (RARRAY_LEN(src_values) != RARRAY_LEN(dst_values)) {
    rb_raise(rb_eArgError, "Source and destination arrays must have the same length");
  }
  
  std::vector<mx::array> src_arrays;
  std::vector<mx::array> dst_arrays;
  
  for (long i = 0; i < RARRAY_LEN(src_values); i++) {
    VALUE src = rb_ary_entry(src_values, i);
    VALUE dst = rb_ary_entry(dst_values, i);
    
    if (!rb_obj_is_kind_of(src, rb_path2class("MLX::Core::Array")) ||
        !rb_obj_is_kind_of(dst, rb_path2class("MLX::Core::Array"))) {
      rb_raise(rb_eTypeError, "Expected all elements to be MLX::Core::Array objects");
    }
    
    src_arrays.push_back(get_array(src));
    dst_arrays.push_back(get_array(dst));
  }
  
  tree_replace(tree, src_arrays, dst_arrays);
  return tree;
}

// Tree flatten arrays function for Ruby
VALUE rb_tree_flatten_arrays(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE tree = argv[0];
  bool strict = (argc > 1) ? RTEST(argv[1]) : true;
  
  auto flat_arrays = tree_flatten(tree, strict);
  
  VALUE result = rb_ary_new2(flat_arrays.size());
  for (size_t i = 0; i < flat_arrays.size(); i++) {
    rb_ary_store(result, i, wrap_array(flat_arrays[i]));
  }
  
  return result;
}

// Tree flatten with structure function for Ruby
VALUE rb_tree_flatten_with_structure(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE tree = argv[0];
  bool strict = (argc > 1) ? RTEST(argv[1]) : true;
  
  auto [flat_arrays, structure] = tree_flatten_with_structure(tree, strict);
  
  VALUE result = rb_ary_new2(2);
  VALUE array_values = rb_ary_new2(flat_arrays.size());
  
  for (size_t i = 0; i < flat_arrays.size(); i++) {
    rb_ary_store(array_values, i, wrap_array(flat_arrays[i]));
  }
  
  rb_ary_store(result, 0, array_values);
  rb_ary_store(result, 1, structure);
  
  return result;
}

// Tree unflatten from structure function for Ruby
VALUE rb_tree_unflatten_from_structure(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE structure = argv[0];
  VALUE arrays = argv[1];
  int start_index = (argc > 2) ? NUM2INT(argv[2]) : 0;
  
  Check_Type(arrays, T_ARRAY);
  
  std::vector<mx::array> cpp_arrays;
  for (long i = 0; i < RARRAY_LEN(arrays); i++) {
    VALUE arr = rb_ary_entry(arrays, i);
    if (!rb_obj_is_kind_of(arr, rb_path2class("MLX::Core::Array"))) {
      rb_raise(rb_eTypeError, "Expected all elements to be MLX::Core::Array objects");
    }
    cpp_arrays.push_back(get_array(arr));
  }
  
  return tree_unflatten_from_structure(structure, cpp_arrays, start_index);
}

// Initialize trees module
void init_trees(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "tree_flatten", RUBY_METHOD_FUNC(rb_tree_flatten), 1);
  rb_define_module_function(module, "tree_unflatten", RUBY_METHOD_FUNC(rb_tree_unflatten), 2);
  rb_define_module_function(module, "tree_map", RUBY_METHOD_FUNC(rb_tree_map), 2);
  rb_define_module_function(module, "tree_fill", RUBY_METHOD_FUNC(rb_tree_fill), 2);
  rb_define_module_function(module, "tree_replace", RUBY_METHOD_FUNC(rb_tree_replace), 3);
  rb_define_module_function(module, "tree_flatten_arrays", RUBY_METHOD_FUNC(rb_tree_flatten_arrays), -1);
  rb_define_module_function(module, "tree_flatten_with_structure", RUBY_METHOD_FUNC(rb_tree_flatten_with_structure), -1);
  rb_define_module_function(module, "tree_unflatten_from_structure", RUBY_METHOD_FUNC(rb_tree_unflatten_from_structure), -1);
} 