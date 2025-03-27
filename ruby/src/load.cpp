#include <ruby.h>
#include <unordered_map>
#include <string>
#include <memory>
#include <fstream>
#include <stdexcept>
#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "ruby/src/load.h"

namespace mx = mlx::core;

// Define custom types since MLX doesn't expose these
namespace mlx::core {

// Custom metadata type for GGUF
using Metadata = std::unordered_map<std::string, std::string>;

// Stub for GGUFMetadata variant if needed
struct GGUFMetaData {
  std::variant<std::string, int64_t, double, bool, 
               std::vector<std::string>, std::vector<int64_t>, 
               std::vector<double>, std::vector<bool>> data;
  
  // Constructors for different types
  GGUFMetaData(const std::string& s) : data(s) {}
  GGUFMetaData(int64_t i) : data(i) {}
  GGUFMetaData(double d) : data(d) {}
  GGUFMetaData(bool b) : data(b) {}
  GGUFMetaData(const std::vector<std::string>& v) : data(v) {}
  GGUFMetaData(const std::vector<int64_t>& v) : data(v) {}
  GGUFMetaData(const std::vector<double>& v) : data(v) {}
  
  // Default constructor required for std::unordered_map
  GGUFMetaData() : data(std::string("")) {}
};

// Stub for GGUF loading result
struct GGUFLoad {
  std::unordered_map<std::string, array> tensors;
  std::unordered_map<std::string, std::string> metadata;
};

// Helper for safetensors load result
// Python returns: pair<unordered_map<string, array>, unordered_map<string, string>>
// We can store that in Ruby as [hash_of_arrays, hash_of_metadata].
inline std::pair<std::unordered_map<std::string, array>,
                 std::unordered_map<std::string, std::string>>
load_safetensors_to_map(const std::string& file,
                        StreamOrDevice s) {
  // In actual code, you'd call:
  //   return load_safetensors(file, s);
  // For demonstration, returning a stub with empty containers.
  std::unordered_map<std::string, array> arrays;
  std::unordered_map<std::string, std::string> metadata;
  // ...
  return {arrays, metadata};
}

} // namespace mlx::core

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

// Helper to extract Stream or Device from Ruby VALUE
static mx::StreamOrDevice get_stream_or_device(VALUE obj) {
  if (NIL_P(obj)) {
    return mx::StreamOrDevice{}; // Default empty stream/device
  }
  
  // Check if it's a Stream object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Stream"))) {
    mx::Stream* stream_ptr;
    Data_Get_Struct(obj, mx::Stream, stream_ptr);
    return *stream_ptr;
  }
  
  // Check if it's a Device object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Device"))) {
    mx::Device* device_ptr;
    Data_Get_Struct(obj, mx::Device, device_ptr);
    return *device_ptr;
  }
  
  rb_raise(rb_eTypeError, "Expected Stream or Device object");
  return mx::StreamOrDevice{}; // Never reached
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
    
    // Create a copy of the array to insert into the map
    result.insert(std::make_pair(key_str, *value_arr));
  }
  
  return result;
}

// Helper to convert Ruby Hash to C++ unordered_map<string, string>
static std::unordered_map<std::string, std::string> ruby_hash_to_string_map(VALUE hash) {
  Check_Type(hash, T_HASH);
  
  std::unordered_map<std::string, std::string> result;
  
  VALUE keys = rb_funcall(hash, rb_intern("keys"), 0);
  for (long i = 0; i < RARRAY_LEN(keys); i++) {
    VALUE key = rb_ary_entry(keys, i);
    VALUE value = rb_hash_aref(hash, key);
    
    // Convert key and value to string
    std::string key_str = StringValueCStr(key);
    std::string value_str = StringValueCStr(value);
    
    result.insert(std::make_pair(key_str, value_str));
  }
  
  return result;
}

// Helper to convert Ruby Hash to C++ unordered_map<string, GGUFMetaData>
static std::unordered_map<std::string, mx::GGUFMetaData> ruby_hash_to_gguf_metadata(VALUE hash) {
  Check_Type(hash, T_HASH);
  
  std::unordered_map<std::string, mx::GGUFMetaData> result;
  
  VALUE keys = rb_funcall(hash, rb_intern("keys"), 0);
  for (long i = 0; i < RARRAY_LEN(keys); i++) {
    VALUE key = rb_ary_entry(keys, i);
    VALUE value = rb_hash_aref(hash, key);
    
    // Convert key to string
    std::string key_str = StringValueCStr(key);
    
    // Handle different types of metadata
    if (RB_TYPE_P(value, T_STRING)) {
      result.insert(std::make_pair(key_str, mx::GGUFMetaData(std::string(StringValueCStr(value)))));
    } else if (RB_TYPE_P(value, T_FIXNUM)) {
      result.insert(std::make_pair(key_str, mx::GGUFMetaData(NUM2LL(value))));
    } else if (RB_TYPE_P(value, T_FLOAT)) {
      result.insert(std::make_pair(key_str, mx::GGUFMetaData(NUM2DBL(value))));
    } else if (RB_TYPE_P(value, T_TRUE) || RB_TYPE_P(value, T_FALSE)) {
      result.insert(std::make_pair(key_str, mx::GGUFMetaData(RTEST(value))));
    } else if (RB_TYPE_P(value, T_ARRAY)) {
      // Determine array type based on first element
      if (RARRAY_LEN(value) > 0) {
        VALUE first = rb_ary_entry(value, 0);
        if (RB_TYPE_P(first, T_STRING)) {
          std::vector<std::string> str_vec;
          for (long j = 0; j < RARRAY_LEN(value); j++) {
            VALUE item = rb_ary_entry(value, j);
            str_vec.push_back(StringValueCStr(item));
          }
          result.insert(std::make_pair(key_str, mx::GGUFMetaData(str_vec)));
        } else if (RB_TYPE_P(first, T_FIXNUM)) {
          std::vector<int64_t> int_vec;
          for (long j = 0; j < RARRAY_LEN(value); j++) {
            VALUE item = rb_ary_entry(value, j);
            int_vec.push_back(NUM2LL(item));
          }
          result.insert(std::make_pair(key_str, mx::GGUFMetaData(int_vec)));
        } else if (RB_TYPE_P(first, T_FLOAT)) {
          std::vector<double> float_vec;
          for (long j = 0; j < RARRAY_LEN(value); j++) {
            VALUE item = rb_ary_entry(value, j);
            float_vec.push_back(NUM2DBL(item));
          }
          result.insert(std::make_pair(key_str, mx::GGUFMetaData(float_vec)));
        }
      }
    }
  }
  
  return result;
}

// Helper to convert C++ unordered_map<string, array> to Ruby Hash
static VALUE map_to_ruby_hash(const std::unordered_map<std::string, mx::array>& map) {
  VALUE result = rb_hash_new();
  
  for (const auto& [key, value] : map) {
    VALUE rb_key = rb_str_new_cstr(key.c_str());
    VALUE rb_value = wrap_array(value);
    
    rb_hash_aset(result, rb_key, rb_value);
  }
  
  return result;
}

// Helper to convert C++ unordered_map<string, string> to Ruby Hash
static VALUE metadata_to_ruby_hash(const std::unordered_map<std::string, std::string>& metadata) {
  VALUE result = rb_hash_new();
  
  for (const auto& [key, value] : metadata) {
    VALUE rb_key = rb_str_new_cstr(key.c_str());
    VALUE rb_value = rb_str_new_cstr(value.c_str());
    
    rb_hash_aset(result, rb_key, rb_value);
  }
  
  return result;
}

// Ruby IO adapter for MLX IO operations
class RubyFileReader : public mx::io::Reader {
public:
  RubyFileReader(VALUE file) : file_(file) {
    // Ensure file has necessary methods
    if (!rb_respond_to(file, rb_intern("read")) || 
        !rb_respond_to(file, rb_intern("seek")) || 
        !rb_respond_to(file, rb_intern("tell"))) {
      rb_raise(rb_eTypeError, "File object must respond to read, seek, and tell");
    }
  }
  
  ~RubyFileReader() {
    // No explicit cleanup needed for Ruby objects due to GC
  }
  
  bool is_open() const override {
    // Check if the file is closed
    if (rb_respond_to(file_, rb_intern("closed?"))) {
      return !RTEST(rb_funcall(file_, rb_intern("closed?"), 0));
    }
    return true; // Assume open if no "closed?" method
  }
  
  bool good() const override {
    return !NIL_P(file_) && is_open();
  }
  
  size_t tell() override {
    return NUM2ULL(rb_funcall(file_, rb_intern("tell"), 0));
  }
  
  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg) override {
    int whence = 0; // SEEK_SET
    if (way == std::ios_base::cur) {
      whence = 1; // SEEK_CUR
    } else if (way == std::ios_base::end) {
      whence = 2; // SEEK_END
    }
    rb_funcall(file_, rb_intern("seek"), 2, ULL2NUM(off), INT2FIX(whence));
  }
  
  void read(char* data, size_t n) override {
    VALUE buffer = rb_funcall(file_, rb_intern("read"), 1, ULL2NUM(n));
    if (NIL_P(buffer) || RSTRING_LEN(buffer) < n) {
      throw std::runtime_error("Failed to read from Ruby IO stream");
    }
    std::memcpy(data, RSTRING_PTR(buffer), n);
  }
  
  void read(char* data, size_t n, size_t offset) override {
    seek(offset, std::ios_base::beg);
    read(data, n);
  }
  
  std::string label() const override {
    return "Ruby IO object";
  }

private:
  VALUE file_; // Ruby IO object
};

// Ruby IO adapter for MLX IO writing operations
class RubyFileWriter : public mx::io::Writer {
public:
  RubyFileWriter(VALUE file) : file_(file) {
    // Ensure file has necessary methods
    if (!rb_respond_to(file, rb_intern("write")) || 
        !rb_respond_to(file, rb_intern("seek")) || 
        !rb_respond_to(file, rb_intern("tell"))) {
      rb_raise(rb_eTypeError, "File object must respond to write, seek, and tell");
    }
  }
  
  ~RubyFileWriter() {
    // No explicit cleanup needed for Ruby objects due to GC
  }
  
  bool is_open() const override {
    // Check if the file is closed
    if (rb_respond_to(file_, rb_intern("closed?"))) {
      return !RTEST(rb_funcall(file_, rb_intern("closed?"), 0));
    }
    return true; // Assume open if no "closed?" method
  }
  
  bool good() const override {
    return !NIL_P(file_) && is_open();
  }
  
  size_t tell() override {
    return NUM2ULL(rb_funcall(file_, rb_intern("tell"), 0));
  }
  
  void seek(int64_t off, std::ios_base::seekdir way = std::ios_base::beg) override {
    int whence = 0; // SEEK_SET
    if (way == std::ios_base::cur) {
      whence = 1; // SEEK_CUR
    } else if (way == std::ios_base::end) {
      whence = 2; // SEEK_END
    }
    rb_funcall(file_, rb_intern("seek"), 2, ULL2NUM(off), INT2FIX(whence));
  }
  
  void write(const char* data, size_t n) override {
    VALUE buffer = rb_str_new(data, n);
    VALUE bytes_written = rb_funcall(file_, rb_intern("write"), 1, buffer);
    
    if (NIL_P(bytes_written) || NUM2ULL(bytes_written) < n) {
      throw std::runtime_error("Failed to write to Ruby IO stream");
    }
  }
  
  std::string label() const override {
    return "Ruby IO object";
  }

private:
  VALUE file_; // Ruby IO object
};

// Helper function to check if a Ruby object is a file-like object for reading
static bool is_readable_file(VALUE obj) {
  return rb_respond_to(obj, rb_intern("read")) && 
         rb_respond_to(obj, rb_intern("seek")) && 
         rb_respond_to(obj, rb_intern("tell"));
}

// Helper function to check if a Ruby object is a file-like object for writing
static bool is_writable_file(VALUE obj) {
  return rb_respond_to(obj, rb_intern("write")) && 
         rb_respond_to(obj, rb_intern("seek")) && 
         rb_respond_to(obj, rb_intern("tell"));
}

// Helper function to check if a file is a zip file
static bool is_zip_file(VALUE file) {
  // Load zipfile module from Ruby
  VALUE zipfile_module = rb_const_get(rb_cObject, rb_intern("Zip"));
  
  if (RB_TYPE_P(file, T_STRING)) {
    // For string paths, use Zip.zip_file? method
    return RTEST(rb_funcall(zipfile_module, rb_intern("zip_file?"), 1, file));
  } else if (is_readable_file(file)) {
    // For file objects, use Zip.zip_file? method and restore position
    VALUE pos = rb_funcall(file, rb_intern("tell"), 0);
    VALUE result = rb_funcall(zipfile_module, rb_intern("zip_file?"), 1, file);
    rb_funcall(file, rb_intern("seek"), 2, pos, INT2FIX(0));
    return RTEST(result);
  }
  
  return false;
}

// Load module methods
static VALUE load_load(int argc, VALUE* argv, VALUE self) {
  // We want up to 4 arguments: (file, format, return_metadata, stream)
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }

  VALUE file        = argv[0];
  VALUE format_val  = (argc >= 2) ? argv[1] : Qnil;
  VALUE meta_val    = (argc >= 3) ? argv[2] : Qnil;
  VALUE stream_val  = (argc >= 4) ? argv[3] : Qnil;

  // Convert optional arguments
  std::optional<std::string> format;
  bool return_metadata = false;
  if (!NIL_P(format_val)) {
    format = std::string(StringValueCStr(format_val));
  }
  if (!NIL_P(meta_val)) {
    return_metadata = RTEST(meta_val);
  }
  mx::StreamOrDevice s = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);

  // If format is not provided, we can attempt extension-based guess or fallback
  if (!format.has_value()) {
    if (RB_TYPE_P(file, T_STRING)) {
      std::string fname = StringValueCStr(file);
      // naive extension parse
      auto pos = fname.find_last_of('.');
      if (pos == std::string::npos) {
        rb_raise(rb_eArgError, "[load] Could not infer file format from extension");
      }
      format = fname.substr(pos+1);
    } else {
      // For Ruby IO, we can't trivially guess. Python code tries file.name if available
      rb_raise(rb_eArgError, "[load] Must specify format explicitly when passing IO");
    }
  }

  // Now route to correct loader
  std::string f = *format;
  if (f == "npy") {
    // For .npy, metadata not supported
    if (return_metadata) {
      rb_raise(rb_eArgError, "[load] return_metadata not supported for npy/npz");
    }
    return load_load_npy(argc, argv, self); // or call the direct helper
  } else if (f == "npz") {
    if (return_metadata) {
      rb_raise(rb_eArgError, "[load] return_metadata not supported for npz");
    }
    // We can just do same approach as python: call load_load_npz
    VALUE args2[2] = { file, stream_val };
    return load_load_npz(2, args2, self);
  } else if (f == "safetensors") {
    // Use load_safetensors
    VALUE args2[2] = { file, stream_val };
    VALUE safetensor_res = load_load_safetensors(2, args2, self);
    if (!return_metadata) {
      // If we aren't returning metadata, we just return the first part
      // i.e. the dictionary of arrays
      return rb_ary_entry(safetensor_res, 0);
    } else {
      return safetensor_res; // Already a 2-element array [tensor_map, metadata_map]
    }
  } else if (f == "gguf") {
    // Use load_gguf
    VALUE args2[2] = { file, stream_val };
    VALUE gguf_res = load_load_gguf(2, args2, self);
    if (!return_metadata) {
      // first part of returned array
      return rb_ary_entry(gguf_res, 0);
    } else {
      // if gguf_res is [tensors_hash, metadata_hash], return it
      return gguf_res;
    }
  }
  rb_raise(rb_eArgError, "[load] Unknown file format '%s'", f.c_str());
  return Qnil; // unreachable
}

static VALUE load_load_shard(int argc, VALUE* argv, VALUE self) {
  if (argc < 3 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 3..4)", argc);
  }
  
  rb_raise(rb_eNotImpError, "The load_shard function is not yet implemented");
  return Qnil;
}

static VALUE load_save(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }

  // In Python, 'save' is a simplified helper that picks format by extension or explicit...
  // For now, we can do the same trick as 'load' if you want a direct "save" to guess format.
  rb_raise(rb_eNotImpError, "General 'save' not implemented (analog to python's load).");
  return Qnil;
}

static VALUE load_save_shard(int argc, VALUE* argv, VALUE self) {
  if (argc < 4 || argc > 5) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 4..5)", argc);
  }
  
  rb_raise(rb_eNotImpError, "The save_shard function is not yet implemented");
  return Qnil;
}

static VALUE load_load_safetensors(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  VALUE file_val   = argv[0];
  VALUE stream_val = (argc == 2) ? argv[1] : Qnil;
  mx::StreamOrDevice s = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);

  // In Python, load_safetensors can return (dict, metadata)
  // Here, we can mimic that by calling a helper
  // For demonstration, call a stub:
  std::pair<std::unordered_map<std::string, mx::array>,
            std::unordered_map<std::string, std::string>> result;

  if (RB_TYPE_P(file_val, T_STRING)) {
    std::string path_str = StringValueCStr(file_val);
    result = mx::core::load_safetensors_to_map(path_str, s);
  } else {
    rb_raise(rb_eTypeError, "[load_safetensors] Only String filenames are currently supported");
  }

  // Convert the map of arrays and map of metadata to Ruby
  VALUE arrays_hash   = map_to_ruby_hash(result.first);
  VALUE metadata_hash = metadata_to_ruby_hash(result.second);

  // Return them as an array [arrays_hash, metadata_hash], just like python returns a tuple
  VALUE ret = rb_ary_new2(2);
  rb_ary_store(ret, 0, arrays_hash);
  rb_ary_store(ret, 1, metadata_hash);
  return ret;
}

static VALUE load_save_safetensors(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }

  // In Python: save_safetensors(file, dict_of_arrays, metadata_dict=None)
  // We'll do the same approach here
  VALUE file_val   = argv[0];
  VALUE map_val    = argv[1];
  VALUE meta_val   = (argc >= 3) ? argv[2] : Qnil;
  VALUE stream_val = (argc == 4) ? argv[3] : Qnil; // if you want an optional stream

  // For parallelism with Python, we ignore stream here, or raise if we like
  if (!NIL_P(stream_val)) {
    // Possibly do something with stream?
  }

  std::unordered_map<std::string, mx::array> arrays_map = ruby_hash_to_map(map_val);
  std::unordered_map<std::string, std::string> metadata_map;
  if (!NIL_P(meta_val)) {
    metadata_map = ruby_hash_to_string_map(meta_val);
  }

  // In the real code, you'd call: mx::save_safetensors(path, arrays_map, metadata_map);
  if (RB_TYPE_P(file_val, T_STRING)) {
    std::string path_str = StringValueCStr(file_val);
    
    // Add .npy to file name if it is not there
    if (path_str.length() < 4 || path_str.substr(path_str.length() - 4, 4) != ".npy") {
      path_str += ".npy";
    }
    
    // Use MLX's NPY save functionality directly
    try {
      // In a real implementation, we would save the array to a NPY file here
      std::ofstream out_file(path_str, std::ios::binary);
      if (!out_file.is_open()) {
        rb_raise(rb_eArgError, "Could not open file for writing: %s", path_str.c_str());
        return Qnil;
      }
      
      // For now, we just evaluate the array to ensure all operations are complete
      for (auto& kv : arrays_map) {
        kv.second.eval();
      }
    } catch (const std::exception& e) {
    // For demonstration, do nothing or check opening
    std::ofstream out_file(path_str, std::ios::binary);
    if(!out_file.is_open()) {
      rb_raise(rb_eArgError, "Could not open file for safetensors: %s", path_str.c_str());
    }
    // Evaluate arrays
    for (auto& kv : arrays_map) {
      kv.second.eval();
    }
    // Stub: would actually call mx::save_safetensors(...)
  } else if (is_writable_file(file_val)) {
    // handle as stream
    auto writer = std::make_shared<RubyFileWriter>(file_val);
    // stub: would call mx::save_safetensors(writer, arrays_map, metadata_map);
  } else {
    rb_raise(rb_eTypeError, "[save_safetensors] Expected String or IO-like object");
  }

  return Qnil;
}

static VALUE load_load_gguf(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  VALUE file_val   = argv[0];
  VALUE stream_val = (argc == 2) ? argv[1] : Qnil;
  mx::StreamOrDevice s = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);

  // The Python returns (weightsMap, metadataMap) if requested
  // We'll do it unconditionally here, returning [weightsHash, metadataHash].
  // In real code, you would do:
  //   auto result = mx::load_gguf(filename, s);
  //   auto &weights = result.first;
  //   auto &metadata = result.second;
  // We'll just stub it:
  std::unordered_map<std::string, mx::array> weights;
  std::unordered_map<std::string, std::string> metadata;

  // Convert to Ruby
  VALUE weights_hash = map_to_ruby_hash(weights);
  VALUE meta_hash    = metadata_to_ruby_hash(metadata);

  VALUE ret = rb_ary_new2(2);
  rb_ary_store(ret, 0, weights_hash);
  rb_ary_store(ret, 1, meta_hash);
  return ret;
}

static VALUE load_save_gguf(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  // In Python: save_gguf(file, dict_of_arrays, metadata_dict=None)
  VALUE file_val   = argv[0];
  VALUE map_val    = argv[1];
  VALUE meta_val   = (argc >= 3) ? argv[2] : Qnil;
  VALUE stream_val = (argc == 4) ? argv[3] : Qnil;

  if (!NIL_P(stream_val)) {
    // optional stream
  }

  std::unordered_map<std::string, mx::array> arrays_map = ruby_hash_to_map(map_val);
  std::unordered_map<std::string, mx::GGUFMetaData> metadata_map;
  if (!NIL_P(meta_val)) {
    metadata_map = ruby_hash_to_gguf_metadata(meta_val);
  }

  // Actually call the MLX function in real code:
  //   mx::save_gguf(file, arrays_map, metadata_map);

  if (RB_TYPE_P(file_val, T_STRING)) {
    std::string path_str = StringValueCStr(file_val);
    std::ofstream out_file(path_str, std::ios::binary);
    if (!out_file.is_open()) {
      rb_raise(rb_eArgError, "Could not open file for gguf: %s", path_str.c_str());
    }
    // Evaluate arrays
    for (auto& kv : arrays_map) {
      kv.second.eval();
    }
  } else {
    // Python code only allows string paths for gguf; do the same here
    rb_raise(rb_eTypeError, "[save_gguf] Input must be a string, not IO");
  }
  return Qnil;
}

// Helper to check if path has .npz extension
static std::string ensure_npz_extension(VALUE path) {
  std::string path_str = StringValueCStr(path);
  
  // Add .npz to file name if it is not there
  if (path_str.length() < 4 || path_str.substr(path_str.length() - 4, 4) != ".npz") {
    path_str += ".npz";
  }
  
  return path_str;
}

static VALUE load_savez(int argc, VALUE* argv, VALUE self) {
  if (argc < 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected at least 1)", argc);
  }
  // The first arg is file, the rest can be arrays or key=>array from a Ruby Hash
  // In Python, we do something like mlx_savez_helper(file, args, kwargs, compressed=false)
  // For demonstration, we'll do a stub:
  VALUE file_val = argv[0];
  // In real code, parse the rest as array or keyword hash.
  rb_warn("savez not yet fully implemented. Stub only.");
  return Qnil;
}

static VALUE load_savez_compressed(int argc, VALUE* argv, VALUE self) {
  if (argc < 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected at least 1)", argc);
  }
  rb_warn("savez_compressed not yet fully implemented. Stub only.");
  return Qnil;
}

static VALUE load_load_npz(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  VALUE file_val   = argv[0];
  VALUE stream_val = (argc == 2) ? argv[1] : Qnil;
  mx::StreamOrDevice s = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);

  // In Python, load_npz returns std::unordered_map<std::string, mx::array>.
  // For demonstration, we will just return an empty hash or a stub.
  // In real code, you'd do something like:
  //   auto dict = mx::load_npz(filename_or_stream, s);

  std::unordered_map<std::string, mx::array> dict;

  // Convert to Ruby
  VALUE rb_dict = map_to_ruby_hash(dict);
  return rb_dict;
}

static VALUE load_load_npy(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE path_or_file = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  // Initialize result using a proper constructor with shape and dtype
  mx::array result = mx::zeros({1}, mx::float32, stream);
  
  if (RB_TYPE_P(path_or_file, T_STRING)) {
    std::string path_str = StringValueCStr(path_or_file);
    // In actual code, you'd do:
    //   result = mx::load(path_str, stream);
    try {
      result = mx::load(path_str, stream); // Real call to MLX
      result.eval();
    } catch (const std::exception& e) {
      rb_raise(rb_eRuntimeError, "Error loading NPY file: %s", e.what());
    }
  } else if (is_readable_file(path_or_file)) {
    // Got a Ruby IO-like object
    auto reader = std::make_shared<RubyFileReader>(path_or_file);
    try {
      // In actual code: result = mx::load(reader, stream);
      result = mx::load(reader, stream);
      result.eval();  // Evaluate immediately
    } catch (const std::exception& e) {
      rb_raise(rb_eRuntimeError, "Error loading NPY file: %s", e.what());
    }
  } else {
    rb_raise(rb_eTypeError, "Expected String or IO-like object");
  }
  
  return wrap_array(result);
}

static VALUE load_save_npy(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE path_or_file = argv[0];
  VALUE array_val = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  mx::array& arr = get_array(array_val);
  
  if (RB_TYPE_P(path_or_file, T_STRING)) {
    std::string path_str = StringValueCStr(path_or_file);
    if (path_str.length() < 4 || path_str.substr(path_str.length() - 4, 4) != ".npy") {
      path_str += ".npy";
    }
    try {
      // Use MLX to actually save the array:
      //   mx::save(path_str, arr);
      std::ofstream ofs(path_str, std::ios::binary);
      if (!ofs.is_open()) {
        rb_raise(rb_eArgError, "Could not open output file: %s", path_str.c_str());
      }
      {
        // If the MLX library has a direct overload:
        //   mx::save(path_str, arr);
        // or we do:
        //   mx::save(ofs, arr);
        // For demonstration:
        //   arr.eval();
      }
      arr.eval();
    } catch (const std::exception& e) {
      rb_raise(rb_eRuntimeError, "Error saving NPY file: %s", e.what());
    }
  } else if (is_writable_file(path_or_file)) {
    // Got a Ruby IO-like object
    auto writer = std::make_shared<RubyFileWriter>(path_or_file);
    
    // Save the array
    try {
      // In a real implementation, we would save the array to the writer here
      arr.eval();
      // Real call:
      //   mx::save(writer, arr);
    } catch (const std::exception& e) {
      rb_raise(rb_eRuntimeError, "Error saving NPY file: %s", e.what());
    }
  } else {
    rb_raise(rb_eTypeError, "Expected String or IO-like object");
  }
  
  return Qnil;
}

// Initialize load module
void init_load(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "load", RUBY_METHOD_FUNC(load_load), -1);
  rb_define_module_function(module, "load_shard", RUBY_METHOD_FUNC(load_load_shard), -1);
  rb_define_module_function(module, "save", RUBY_METHOD_FUNC(load_save), -1);
  rb_define_module_function(module, "save_shard", RUBY_METHOD_FUNC(load_save_shard), -1);
  rb_define_module_function(module, "load_safetensors", RUBY_METHOD_FUNC(load_load_safetensors), -1);
  rb_define_module_function(module, "save_safetensors", RUBY_METHOD_FUNC(load_save_safetensors), -1);
  rb_define_module_function(module, "load_gguf", RUBY_METHOD_FUNC(load_load_gguf), -1);
  rb_define_module_function(module, "save_gguf", RUBY_METHOD_FUNC(load_save_gguf), -1);
  rb_define_module_function(module, "load_npy", RUBY_METHOD_FUNC(load_load_npy), -1);
  rb_define_module_function(module, "save_npy", RUBY_METHOD_FUNC(load_save_npy), -1);
  rb_define_module_function(module, "load_npz", RUBY_METHOD_FUNC(load_load_npz), -1);
  rb_define_module_function(module, "savez", RUBY_METHOD_FUNC(load_savez), -1);
  rb_define_module_function(module, "savez_compressed", RUBY_METHOD_FUNC(load_savez_compressed), -1);
} 