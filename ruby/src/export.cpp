#include <ruby.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <map>
#include <fstream>
#include <functional>
#include "mlx/export.h"
#include "mlx/graph_utils.h"
#include "mlx/io.h"

namespace mx = mlx::core;

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Helper to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Helper to make a copy of mx::array to avoid reference issues
static mx::array copy_array(const mx::array& arr) {
  return mx::array(arr); // Use copy constructor
}

// Helper to convert Ruby Hash to C++ unordered_map<string, array>
static std::unordered_map<std::string, mx::array> ruby_hash_to_map(VALUE hash) {
  Check_Type(hash, T_HASH);
  
  std::unordered_map<std::string, mx::array> result;
  
  VALUE keys = rb_funcall(hash, rb_intern("keys"), 0);
  for (long i = 0; i < RARRAY_LEN(keys); i++) {
    VALUE key = rb_ary_entry(keys, i);
    VALUE value = rb_hash_aref(hash, key);
    
    // Convert key to string
    std::string key_str = StringValueCStr(key);
    
    // Extract array from value
    mx::array* value_arr;
    Data_Get_Struct(value, mx::array, value_arr);
    
    // Make a copy to avoid reference issues
    result.emplace(key_str, copy_array(*value_arr));
  }
  
  return result;
}

// Helper to convert Ruby Hash to C++ std::map<string, array>
static std::map<std::string, mx::array> ruby_hash_to_std_map(VALUE hash) {
  Check_Type(hash, T_HASH);
  
  std::map<std::string, mx::array> result;
  
  VALUE keys = rb_funcall(hash, rb_intern("keys"), 0);
  for (long i = 0; i < RARRAY_LEN(keys); i++) {
    VALUE key = rb_ary_entry(keys, i);
    VALUE value = rb_hash_aref(hash, key);
    
    // Convert key to string
    std::string key_str = StringValueCStr(key);
    
    // Extract array from value
    mx::array* value_arr;
    Data_Get_Struct(value, mx::array, value_arr);
    
    // Make a copy to avoid reference issues
    result.emplace(key_str, copy_array(*value_arr));
  }
  
  return result;
}

// Helper to extract arrays from Ruby array
static std::vector<mx::array> ruby_array_to_vector(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  std::vector<mx::array> result;
  for (long i = 0; i < RARRAY_LEN(arr); i++) {
    VALUE item = rb_ary_entry(arr, i);
    mx::array* arr_ptr;
    Data_Get_Struct(item, mx::array, arr_ptr);
    result.push_back(copy_array(*arr_ptr));
  }
  
  return result;
}

// Export module methods
static VALUE export_to_safetensors(VALUE self, VALUE weights, VALUE path, VALUE metadata = Qnil) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  auto weights_map = ruby_hash_to_map(weights);
  
  // Process metadata if provided
  if (metadata != Qnil) {
    // Convert Ruby hash to std::unordered_map<std::string, std::string>
    std::unordered_map<std::string, std::string> metadata_map;
    Check_Type(metadata, T_HASH);
    
    VALUE meta_keys = rb_funcall(metadata, rb_intern("keys"), 0);
    for (long i = 0; i < RARRAY_LEN(meta_keys); i++) {
      VALUE key = rb_ary_entry(meta_keys, i);
      VALUE value = rb_hash_aref(metadata, key);
      
      Check_Type(key, T_STRING);
      Check_Type(value, T_STRING);
      
      std::string key_str = StringValueCStr(key);
      std::string value_str = StringValueCStr(value);
      
      metadata_map[key_str] = value_str;
    }
    
    // Save with metadata
    mx::save_safetensors(path_str, weights_map, metadata_map);
  } else {
    // Save without metadata
    mx::save_safetensors(path_str, weights_map);
  }
  
  return Qnil;
}

static VALUE export_to_gguf(VALUE self, VALUE weights, VALUE path, VALUE metadata = Qnil) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  auto weights_map = ruby_hash_to_map(weights);
  
  if (metadata != Qnil) {
    // Process metadata
    rb_raise(rb_eNotImpError, "GGUF metadata support not yet implemented for Ruby bindings");
    return Qnil;
  }
  
  // Save without metadata
  mx::save_gguf(path_str, weights_map);
  return Qnil;
}

static VALUE export_to_dot(int argc, VALUE* argv, VALUE self) {
  if (argc < 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected at least 2)", argc);
  }
  
  // Extract file parameter
  VALUE file = argv[0];
  Check_Type(file, T_STRING);
  std::string file_path = StringValueCStr(file);
  
  // Gather arrays from arguments
  std::vector<mx::array> arrays;
  mx::NodeNamer namer;
  
  // Process positional array arguments (starting from index 1)
  for (int i = 1; i < argc; i++) {
    VALUE arg = argv[i];
    
    // If it's a hash, treat as keyword arguments with names
    if (RB_TYPE_P(arg, T_HASH)) {
      VALUE keys = rb_funcall(arg, rb_intern("keys"), 0);
      for (long j = 0; j < RARRAY_LEN(keys); j++) {
        VALUE key = rb_ary_entry(keys, j);
        VALUE value = rb_hash_aref(arg, key);
        
        if (rb_obj_is_kind_of(value, rb_path2class("MLX::Core::Array"))) {
          mx::array& arr = get_array(value);
          arrays.push_back(copy_array(arr));
          
          // Set name in namer
          std::string name = StringValueCStr(key);
          namer.set_name(arrays.back(), name);
        }
      }
    } else if (rb_obj_is_kind_of(arg, rb_path2class("MLX::Core::Array"))) {
      // Regular array argument
      mx::array& arr = get_array(arg);
      arrays.push_back(copy_array(arr));
    } else if (RB_TYPE_P(arg, T_ARRAY)) {
      // Array of arrays
      for (long j = 0; j < RARRAY_LEN(arg); j++) {
        VALUE item = rb_ary_entry(arg, j);
        if (rb_obj_is_kind_of(item, rb_path2class("MLX::Core::Array"))) {
          mx::array& arr = get_array(item);
          arrays.push_back(copy_array(arr));
        }
      }
    }
  }
  
  // Check if we have any arrays to export
  if (arrays.empty()) {
    rb_raise(rb_eArgError, "No arrays provided for export_to_dot");
  }
  
  // Export to DOT format
  std::ofstream out(file_path);
  if (!out.is_open()) {
    rb_raise(rb_eRuntimeError, "Failed to open file for writing: %s", file_path.c_str());
  }
  
  mx::export_to_dot(out, std::move(namer), arrays);
  
  return Qnil;
}

/*
 * The Python code provides:
 *   def export_function(file, fun, *args, shapeless=False, **kwargs)
 * Here we replicate that in Ruby. For simplicity, we do not fully implement
 * the same "validate_and_extract_inputs" logic. Instead, we parse straightforwardly:
 *
 *   export_function("somefile.mlxfn", ruby_fun, [arg1, arg2], { kw: kwarg })
 *   # shapeless via kwargs
 *
 * If you need 100% identical argument heuristics (tuple-of-arrays, dict-of-arrays, etc.)
 * replicate or port the same conditional logic from Python.
 */

static VALUE rb_export_function(int argc, VALUE *argv, VALUE self)
{
  // Expected Ruby call: export_function(file, fun, *args, shapeless: false, **kwargs)
  // We'll parse it in a simpler way.

  // We'll handle optional named parameters via rb_scan_args and see if
  // the last argument might be a hash with :shapeless and others.

  // For clarity, do an explicit check. A thorough approach would do more error-checking:
  if (argc < 2) {
    rb_raise(rb_eArgError, "export_function requires at least 2 arguments: (file, fun, ...)");
  }

  VALUE file_val = argv[0];
  VALUE fun_val  = argv[1];

  if (TYPE(file_val) != T_STRING) {
    rb_raise(rb_eTypeError, "First argument (file) must be a String");
  }
  std::string file = StringValueCStr(file_val);

  // Check if last argument is a hash that might hold shapeless or kwargs:
  bool shapeless = false;
  std::map<std::string, mx::array> kwargs_map;
  std::vector<mx::array> pos_args;

  // Start scanning positional arrays from argv[2] ... up to the next-to-last if that last is a hash
  int end_pos = argc;
  if (argc > 2 && TYPE(argv[argc - 1]) == T_HASH) {
    // Potentially shapeless or user-specified kwargs in the last argument
    VALUE last_hash = argv[argc - 1];
    VALUE shapeless_val = rb_hash_aref(last_hash, ID2SYM(rb_intern("shapeless")));
    if (shapeless_val != Qnil) {
      shapeless = (shapeless_val == Qtrue);
      rb_hash_delete(last_hash, ID2SYM(rb_intern("shapeless")));
    }
    // treat everything else in last_hash as a kw => array
    VALUE keys = rb_funcall(last_hash, rb_intern("keys"), 0);
    for (long i = 0; i < RARRAY_LEN(keys); i++) {
      VALUE key = rb_ary_entry(keys, i);
      VALUE value = rb_hash_aref(last_hash, key);
      std::string key_str = StringValueCStr(key);
      mx::array &arr = get_array(value);
      kwargs_map.emplace(key_str, copy_array(arr));
    }
    end_pos = argc - 1; // exclude the last from positional scanning
  }

  // Now parse the intermediate positional arguments as arrays
  for (int i = 2; i < end_pos; i++) {
    VALUE val = argv[i];
    // If it's an Array of arrays
    if (TYPE(val) == T_ARRAY) {
      // Expand them
      long len = RARRAY_LEN(val);
      for (long j = 0; j < len; j++) {
        VALUE item = rb_ary_entry(val, j);
        pos_args.push_back(copy_array(get_array(item)));
      }
    } else if (TYPE(val) == T_HASH) {
      // if some user puts a random hash in the middle, you can either raise or parse
      rb_raise(rb_eArgError, "Unexpected Hash in export_function's argument list. Use final kwargs instead.");
    } else {
      // single array
      pos_args.push_back(copy_array(get_array(val)));
    }
  }

  // Now we need to call mx::export_function(...) with a wrapper that calls 'fun_val'.
  // In Python, we do wrap_export_function(fun). In Ruby, we must create a callable that
  // calls fun_val with the correct arguments. We can do that by constructing a std::function
  // that uses rb_funcall.

  // This is the wrapper that will be passed to mx::export_function(...).
  auto wrapper = [fun_val](const std::vector<mx::array> &args_,
                           const std::map<std::string, mx::array> &kwargs_) -> std::vector<mx::array> {
    // We'll call 'fun_val' in Ruby with the same (args_, kwargs_).
    // In Ruby, that's something like: fun_val.call(*wrapped_args, **wrapped_kwargs).

    // Build an array of Ruby arguments
    VALUE ruby_args = rb_ary_new();
    for (auto &a : args_) {
      rb_ary_push(ruby_args, wrap_array(a));
    }
    // Build a Ruby hash for kwargs
    VALUE ruby_kwargs = rb_hash_new();
    for (auto &kv : kwargs_) {
      rb_hash_aset(ruby_kwargs, rb_str_new2(kv.first.c_str()), wrap_array(kv.second));
    }

    // Perform the actual call: fun_val.call(*ruby_args, **ruby_kwargs)
    // In older Ruby C-API, we have to do something more manual for keyword arguments.
    // A simpler fallback approach is fun_val.call(ruby_args, ruby_kwargs).
    // For demonstration, we'll just do fun_val.call(ruby_args, ruby_kwargs).

    VALUE ret = rb_funcall(fun_val, rb_intern("call"), 2, ruby_args, ruby_kwargs);

    // Now we expect either a single mx::array or an array of them
    std::vector<mx::array> outputs;

    if (rb_obj_is_kind_of(ret, rb_path2class("MLX::Core::Array"))) {
      outputs.push_back(copy_array(get_array(ret)));
    } else if (RB_TYPE_P(ret, T_ARRAY)) {
      long len = RARRAY_LEN(ret);
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(ret, i);
        if (!rb_obj_is_kind_of(item, rb_path2class("MLX::Core::Array"))) {
          rb_raise(rb_eArgError, "[export_function wrapper] returned array must contain MLX::Core::Array objects");
        }
        outputs.push_back(copy_array(get_array(item)));
      }
    } else {
      rb_raise(rb_eArgError,
               "[export_function wrapper] function must return an MLX::Core::Array or array of them");
    }
    return outputs;
  };

  // Finally call the underlying MLX export
  mx::export_function(file, wrapper, pos_args, kwargs_map, shapeless);

  return Qnil;
}

/*
 * The Python code has:
 *   def import_function(file: str) -> Callable
 * which returns a callable that, when called, does the underlying imported function.
 * In Ruby, we'll return a Proc or a custom object that can be called the same way.
 */

// Static function for the imported function proc
static VALUE imported_function_proc_handler(int argc, VALUE* argv, VALUE self) {
  // Retrieve the ImportedFunction pointer from the captured data
  // We stored it as a hidden pointer in the Proc's environment
  mx::ImportedFunction* captured_fn;
  Data_Get_Struct(rb_iv_get(self, "@mlx_imported_fn"), mx::ImportedFunction, captured_fn);

  // Next, read the actual call arguments from Ruby
  if (argc < 1) {
    // no arguments: call with empty
    std::vector<mx::array> pos_args;
    std::map<std::string, mx::array> kwargs;
    auto out = (*captured_fn)(pos_args, kwargs);
    // Return a Ruby array
    VALUE ret_ary = rb_ary_new();
    for (auto &x : out) {
      rb_ary_push(ret_ary, wrap_array(x));
    }
    return ret_ary;
  }

  // If last argument is a Hash, treat it as kwargs
  std::map<std::string, mx::array> kwargs;
  if (TYPE(argv[argc-1]) == T_HASH) {
    kwargs = ruby_hash_to_std_map(argv[argc-1]);
    argc--;
  }
  std::vector<mx::array> pos_args;
  for (int i = 0; i < argc; i++) {
    // If an argument is itself an Array or hash, you might replicate Python's logic.
    // For brevity, we treat each one as a single MLX::Core::Array or a T_ARRAY of them.
    if (rb_obj_is_kind_of(argv[i], rb_path2class("MLX::Core::Array"))) {
      pos_args.push_back(copy_array(get_array(argv[i])));
    } else if (RB_TYPE_P(argv[i], T_ARRAY)) {
      // expand
      auto sub = ruby_array_to_vector(argv[i]);
      pos_args.insert(pos_args.end(), sub.begin(), sub.end());
    } else {
      rb_raise(rb_eArgError, "[import_function callable] unsupported argument type");
    }
  }

  auto out = (*captured_fn)(pos_args, kwargs);
  VALUE ret_ary = rb_ary_new();
  for (auto &x : out) {
    rb_ary_push(ret_ary, wrap_array(x));
  }
  return ret_ary; // always return a tuple-of-arrays in Python => an Array in Ruby
}

// Wrapper function with correct rb_block_call_func_t signature
static VALUE imported_function_wrapper(VALUE yielded_arg, VALUE callback_arg, int argc, const VALUE* argv, VALUE blockarg) {
  // Adjust to match the existing handler signature
  return imported_function_proc_handler(argc, const_cast<VALUE*>(argv), callback_arg);
}

static VALUE rb_import_function(VALUE self, VALUE file_val)
{
  if (TYPE(file_val) != T_STRING) {
    rb_raise(rb_eTypeError, "import_function requires a file path String");
  }
  std::string file = StringValueCStr(file_val);

  // Underlying import
  auto fn = mx::import_function(file);

  // Create a Proc with our handler function (using the wrapper with correct signature)
  VALUE proc = rb_proc_new(imported_function_wrapper, Qnil);

  // We must store fn in an allocated structure so it doesn't vanish. We wrap it in a Ruby object:
  mx::ImportedFunction* fn_ptr = new mx::ImportedFunction(fn);
  // attach it to the proc
  rb_iv_set(proc, "@mlx_imported_fn", Data_Wrap_Struct(rb_cObject, NULL, [](void *p) {
    delete reinterpret_cast<mx::ImportedFunction*>(p);
  }, fn_ptr));

  return proc;
}

/*
 * The Python code has a class FunctionExporter with methods close, __enter__, __exit__, __call__.
 * We'll replicate that as a Ruby class. We also have an 'exporter' function returning it.
 */

// This struct will store an mx::FunctionExporter plus the original Ruby fun object (to keep GC references).
typedef struct {
  // Use a pointer to mx::FunctionExporter since we can't default construct it
  std::unique_ptr<mx::FunctionExporter> exporter;
  VALUE ruby_fun; // so GC doesn't free it
} RBFunctionExporter;

static void rb_function_exporter_free(void *ptr) {
  delete reinterpret_cast<RBFunctionExporter*>(ptr);
}

static VALUE rb_function_exporter_alloc(VALUE klass) {
  // Just allocate the struct without initializing exporter
  RBFunctionExporter* data = new RBFunctionExporter();
  data->exporter = nullptr; // Will be initialized in initialize
  data->ruby_fun = Qnil;
  return Data_Wrap_Struct(klass, NULL, rb_function_exporter_free, data);
}

static VALUE rb_function_exporter_initialize(int argc, VALUE *argv, VALUE self) {
  // in Python: def __init__(file, fun, shapeless=False)
  // We do basically the same: exporter(file, fun, shapeless: false)
  // but for direct creation, we parse now. Typically you'd call exporter(...) to get it.
  if (argc < 2) {
    rb_raise(rb_eArgError, "FunctionExporter.new requires at least (file, fun)");
  }
  VALUE file_val = argv[0];
  VALUE fun_val  = argv[1];
  bool shapeless = false;
  if (argc > 2 && TYPE(argv[2]) == T_HASH) {
    VALUE s = rb_hash_aref(argv[2], ID2SYM(rb_intern("shapeless")));
    if (s == Qtrue) {
      shapeless = true;
    }
  }
  if (TYPE(file_val) != T_STRING) {
    rb_raise(rb_eTypeError, "First argument must be String path");
  }

  // get C++ struct
  RBFunctionExporter* data;
  Data_Get_Struct(self, RBFunctionExporter, data);
  data->ruby_fun = fun_val;
  rb_gc_register_address(&data->ruby_fun);

  std::string file = StringValueCStr(file_val);

  // build wrapper function for the exporter:
  auto wrapper = [fun_val](const std::vector<mx::array> &args_,
                           const std::map<std::string, mx::array> &kwargs_) -> std::vector<mx::array> {
    // same approach as export_function wrapper above
    VALUE ruby_args = rb_ary_new();
    for (auto &a : args_) {
      rb_ary_push(ruby_args, wrap_array(a));
    }
    VALUE ruby_kwargs = rb_hash_new();
    for (auto &kv : kwargs_) {
      rb_hash_aset(ruby_kwargs, rb_str_new2(kv.first.c_str()), wrap_array(kv.second));
    }
    VALUE ret = rb_funcall(fun_val, rb_intern("call"), 2, ruby_args, ruby_kwargs);

    std::vector<mx::array> outputs;
    if (rb_obj_is_kind_of(ret, rb_path2class("MLX::Core::Array"))) {
      outputs.push_back(copy_array(get_array(ret)));
    } else if (RB_TYPE_P(ret, T_ARRAY)) {
      long len = RARRAY_LEN(ret);
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(ret, i);
        if (!rb_obj_is_kind_of(item, rb_path2class("MLX::Core::Array"))) {
          rb_raise(rb_eArgError, "[FunctionExporter] returned array must have MLX::Core::Array elements");
        }
        outputs.push_back(copy_array(get_array(item)));
      }
    } else {
      rb_raise(rb_eArgError, "[FunctionExporter] function must return an array or array-of-arrays");
    }
    return outputs;
  };

  // Create the exporter using the constructor directly
  data->exporter = std::make_unique<mx::FunctionExporter>(mx::exporter(file, wrapper, shapeless));
  return self;
}

static VALUE rb_function_exporter_close(VALUE self) {
  RBFunctionExporter* data;
  Data_Get_Struct(self, RBFunctionExporter, data);
  if (data->exporter) {
    data->exporter->close();
  }
  return Qnil;
}

static VALUE rb_function_exporter_call(int argc, VALUE *argv, VALUE self) {
  // Python: __call__(self, *args, **kwargs)
  // We'll parse them quickly again. Then pass to exporter(args, kwargs).
  RBFunctionExporter* data;
  Data_Get_Struct(self, RBFunctionExporter, data);

  if (!data->exporter) {
    rb_raise(rb_eRuntimeError, "FunctionExporter not initialized or already closed");
  }

  // parse as we did in export_function
  std::vector<mx::array> pos_args;
  std::map<std::string, mx::array> kwargs_map;

  if (argc > 0 && TYPE(argv[argc-1]) == T_HASH) {
    kwargs_map = ruby_hash_to_std_map(argv[argc-1]);
    argc--;
  }
  for (int i = 0; i < argc; i++) {
    if (rb_obj_is_kind_of(argv[i], rb_path2class("MLX::Core::Array"))) {
      pos_args.push_back(copy_array(get_array(argv[i])));
    } else if (RB_TYPE_P(argv[i], T_ARRAY)) {
      auto sub = ruby_array_to_vector(argv[i]);
      pos_args.insert(pos_args.end(), sub.begin(), sub.end());
    } else {
      rb_raise(rb_eArgError, "[FunctionExporter#call] unsupported argument type");
    }
  }

  // Now call the underlying exporter
  (*data->exporter)(pos_args, kwargs_map);
  return Qnil;
}

static VALUE rb_function_exporter_enter(VALUE self) {
  // In Python: def __enter__(self): return self
  return self;
}

static VALUE rb_function_exporter_exit(int argc, VALUE *argv, VALUE self) {
  // Python: def __exit__(self, exc_type, exc_value, traceback): self.close()
  rb_function_exporter_close(self);
  return Qnil;
}

/*
 * Now add the exporter(...) function that returns a new FunctionExporter instance.
 * def exporter(file, fun, shapeless: false) -> FunctionExporter
 */
static VALUE rb_exporter(int argc, VALUE *argv, VALUE self) {
  // We'll new-up a FunctionExporter. The same signature as Python:
  //   exporter(file, fun, shapeless=False)
  // we can just call FunctionExporter.new(...) in Ruby:
  VALUE klass = rb_path2class("MLX::Export::FunctionExporter");
  VALUE obj = rb_funcall2(klass, rb_intern("new"), argc, argv);
  return obj;
}

// Initialize export module
void init_export(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "to_safetensors", RUBY_METHOD_FUNC(export_to_safetensors), -1);
  rb_define_module_function(module, "to_gguf", RUBY_METHOD_FUNC(export_to_gguf), -1);
  rb_define_module_function(module, "to_dot", RUBY_METHOD_FUNC(export_to_dot), -1);

  // Also alias "export_to_dot" to match the Python naming (if you want identical name):
  rb_define_alias(module, "export_to_dot", "to_dot");

  // Add missing Python methods:
  rb_define_module_function(module, "export_function", RUBY_METHOD_FUNC(rb_export_function), -1);
  rb_define_module_function(module, "import_function", RUBY_METHOD_FUNC(rb_import_function), 1);
  rb_define_module_function(module, "exporter", RUBY_METHOD_FUNC(rb_exporter), -1);

  // Define the FunctionExporter class inside your module. Python code calls it directly as
  //   from this module import FunctionExporter
  // So in Ruby, you can nest it under e.g. MLX::Export. Adjust if you want it under MLX::Core.
  VALUE cFunctionExporter = rb_define_class_under(module, "FunctionExporter", rb_cObject);
  rb_define_alloc_func(cFunctionExporter, rb_function_exporter_alloc);

  // define the methods that match the Python class interface
  rb_define_method(cFunctionExporter, "initialize",
                   RUBY_METHOD_FUNC(rb_function_exporter_initialize), -1);
  rb_define_method(cFunctionExporter, "close",
                   RUBY_METHOD_FUNC(rb_function_exporter_close), 0);
  rb_define_method(cFunctionExporter, "call",
                   RUBY_METHOD_FUNC(rb_function_exporter_call), -1);
  rb_define_method(cFunctionExporter, "enter",
                   RUBY_METHOD_FUNC(rb_function_exporter_enter), 0);
  rb_define_method(cFunctionExporter, "exit",
                   RUBY_METHOD_FUNC(rb_function_exporter_exit), -1);
} 