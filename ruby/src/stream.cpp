#include <ruby.h>
#include <sstream>
#include "mlx/stream.h"
#include "mlx/device.h"
#include "mlx/utils.h" // Include utils.h for StreamContext and StreamOrDevice

namespace mx = mlx::core;

// Stream class
typedef struct {
    mx::Stream stream;
} StreamWrapper;

// GC callback for Stream
static void stream_free(void* ptr) {
    StreamWrapper* wrapper = static_cast<StreamWrapper*>(ptr);
    ruby_xfree(wrapper);
}

// Helper to extract Stream or Device from Ruby VALUE
static std::variant<std::monostate, mx::Stream, mx::Device> get_stream_or_device(VALUE obj) {
  if (NIL_P(obj)) {
    return std::monostate{}; // Default empty stream/device
  }
  
  // Check if it's a Stream object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Stream"))) {
    StreamWrapper* wrapper;
    Data_Get_Struct(obj, StreamWrapper, wrapper);
    return wrapper->stream;
  }
  
  // Check if it's a Device object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Device"))) {
    mx::Device* device_ptr;
    Data_Get_Struct(obj, mx::Device, device_ptr);
    return *device_ptr;
  }
  
  rb_raise(rb_eTypeError, "Expected Stream or Device object");
  return std::monostate{}; // Never reached
}

// Allocate a new Stream
static VALUE stream_alloc(VALUE klass) {
    StreamWrapper* wrapper = ALLOC(StreamWrapper);
    // Initialize with CPU device's default stream
    wrapper->stream = mx::default_stream(mx::Device::cpu);
    return Data_Wrap_Struct(klass, 0, stream_free, wrapper);
}

// Initialize a Stream with a device
static VALUE stream_initialize(int argc, VALUE* argv, VALUE self) {
    if (argc > 1) {
        rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..1)", argc);
    }
    
    StreamWrapper* wrapper;
    Data_Get_Struct(self, StreamWrapper, wrapper);
    
    if (argc == 1) {
        // Initialize with a device
        VALUE device_obj = argv[0];
        
        if (rb_obj_is_kind_of(device_obj, rb_path2class("MLX::Core::Device"))) {
            mx::Device* device_ptr;
            Data_Get_Struct(device_obj, mx::Device, device_ptr);
            // Create a new stream with the device - set directly
            wrapper->stream = mx::new_stream(*device_ptr);
        } else if (RB_TYPE_P(device_obj, T_FIXNUM)) {
            int device_type = NUM2INT(device_obj);
            mx::Device device(static_cast<mx::Device::DeviceType>(device_type));
            // Create a new stream with the device
            wrapper->stream = mx::new_stream(device);
        } else {
            rb_raise(rb_eTypeError, "Expected Device object or device type integer");
        }
    }
    
    return self;
}

// Get the Stream
static mx::Stream& get_stream(VALUE self) {
    StreamWrapper* wrapper;
    Data_Get_Struct(self, StreamWrapper, wrapper);
    return wrapper->stream;
}

// Stream methods
static VALUE stream_synchronize(VALUE self) {
    mx::Stream& stream = get_stream(self);
    // Stream doesn't have a synchronize method, use the global function
    mx::synchronize(stream);
    return Qnil;
}

static VALUE stream_device(VALUE self) {
    mx::Stream& stream = get_stream(self);
    mx::Device device = stream.device;
    
    VALUE device_class = rb_path2class("MLX::Core::Device");
    mx::Device* device_ptr = new mx::Device(device);
    return Data_Wrap_Struct(device_class, 0, nullptr, device_ptr);
}

static VALUE stream_equal(VALUE self, VALUE other) {
    if (!rb_obj_is_kind_of(other, rb_class_of(self))) {
        return Qfalse;
    }
    
    mx::Stream& self_stream = get_stream(self);
    mx::Stream& other_stream = get_stream(other);
    
    return self_stream == other_stream ? Qtrue : Qfalse;
}

static VALUE stream_inspect(VALUE self) {
    mx::Stream& stream = get_stream(self);
    std::ostringstream os;
    os << "MLX::Core::Stream(index=" << stream.index << ", device=" << stream.device << ")";
    return rb_str_new_cstr(os.str().c_str());
}

// Module functions
static VALUE default_stream(VALUE self, VALUE device_obj) {
    mx::Device device = mx::Device::cpu; // Default to CPU
    
    if (rb_obj_is_kind_of(device_obj, rb_path2class("MLX::Core::Device"))) {
        mx::Device* device_ptr;
        Data_Get_Struct(device_obj, mx::Device, device_ptr);
        device = *device_ptr;
    } else if (RB_TYPE_P(device_obj, T_FIXNUM)) {
        int device_type = NUM2INT(device_obj);
        device = mx::Device(static_cast<mx::Device::DeviceType>(device_type));
    } else {
        rb_raise(rb_eTypeError, "Expected Device object or device type integer");
    }
    
    mx::Stream stream = mx::default_stream(device);
    
    VALUE stream_class = rb_path2class("MLX::Core::Stream");
    StreamWrapper* wrapper = ALLOC(StreamWrapper);
    new (&wrapper->stream) mx::Stream(stream);
    return Data_Wrap_Struct(stream_class, 0, stream_free, wrapper);
}

static VALUE set_default_stream(VALUE self, VALUE stream_obj) {
    if (!rb_obj_is_kind_of(stream_obj, rb_path2class("MLX::Core::Stream"))) {
        rb_raise(rb_eTypeError, "Expected Stream object");
    }
    
    mx::Stream& stream = get_stream(stream_obj);
    mx::set_default_stream(stream);
    
    return Qnil;
}

static VALUE new_stream(VALUE self, VALUE device_obj) {
    mx::Device device = mx::Device::cpu; // Default to CPU
    
    if (rb_obj_is_kind_of(device_obj, rb_path2class("MLX::Core::Device"))) {
        mx::Device* device_ptr;
        Data_Get_Struct(device_obj, mx::Device, device_ptr);
        device = *device_ptr;
    } else if (RB_TYPE_P(device_obj, T_FIXNUM)) {
        int device_type = NUM2INT(device_obj);
        device = mx::Device(static_cast<mx::Device::DeviceType>(device_type));
    } else {
        rb_raise(rb_eTypeError, "Expected Device object or device type integer");
    }
    
    mx::Stream stream = mx::new_stream(device);
    
    VALUE stream_class = rb_path2class("MLX::Core::Stream");
    StreamWrapper* wrapper = ALLOC(StreamWrapper);
    new (&wrapper->stream) mx::Stream(stream);
    return Data_Wrap_Struct(stream_class, 0, stream_free, wrapper);
}

static VALUE synchronize_module(int argc, VALUE* argv, VALUE self) {
    if (argc > 1) {
        rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..1)", argc);
    }
    
    if (argc == 0 || NIL_P(argv[0])) {
        // Synchronize with default stream
        mx::synchronize();
    } else {
        // Synchronize with given stream
        if (!rb_obj_is_kind_of(argv[0], rb_path2class("MLX::Core::Stream"))) {
            rb_raise(rb_eTypeError, "Expected Stream object");
        }
        mx::Stream& stream = get_stream(argv[0]);
        mx::synchronize(stream);
    }
    
    return Qnil;
}

// StreamContext class
typedef struct {
    mx::StreamContext* context;
    VALUE stream_or_device; // Keep a reference to the Ruby object
} StreamContextWrapper;

static void stream_context_free(void* ptr) {
    StreamContextWrapper* wrapper = static_cast<StreamContextWrapper*>(ptr);
    delete wrapper->context;
    ruby_xfree(wrapper);
}

static VALUE stream_context_alloc(VALUE klass) {
    StreamContextWrapper* wrapper = ALLOC(StreamContextWrapper);
    wrapper->context = nullptr;
    wrapper->stream_or_device = Qnil;
    return Data_Wrap_Struct(klass, 0, stream_context_free, wrapper);
}

static VALUE stream_context_initialize(VALUE self, VALUE stream_or_device) {
    StreamContextWrapper* wrapper;
    Data_Get_Struct(self, StreamContextWrapper, wrapper);
    
    // Validate argument - must be Stream or Device
    if (NIL_P(stream_or_device) || 
        (!rb_obj_is_kind_of(stream_or_device, rb_path2class("MLX::Core::Stream")) && 
         !rb_obj_is_kind_of(stream_or_device, rb_path2class("MLX::Core::Device")) &&
         !RB_TYPE_P(stream_or_device, T_FIXNUM))) {
        rb_raise(rb_eArgError, "Invalid argument, please specify a stream or device");
    }
    
    // Store a reference to the Ruby object
    wrapper->stream_or_device = stream_or_device;
    
    return self;
}

static VALUE stream_context_enter(VALUE self) {
    StreamContextWrapper* wrapper;
    Data_Get_Struct(self, StreamContextWrapper, wrapper);
    
    auto s = get_stream_or_device(wrapper->stream_or_device);
    wrapper->context = new mx::StreamContext(s);
    
    return self;
}

static VALUE stream_context_exit(VALUE self, VALUE exc_type, VALUE exc_value, VALUE traceback) {
    StreamContextWrapper* wrapper;
    Data_Get_Struct(self, StreamContextWrapper, wrapper);
    
    if (wrapper->context) {
        delete wrapper->context;
        wrapper->context = nullptr;
    }
    
    return Qnil;
}

// Module stream function - convenience function for StreamContext
static VALUE stream_function(VALUE self, VALUE stream_or_device) {
    // Create a new StreamContext
    VALUE stream_context_class = rb_path2class("MLX::Core::StreamContext");
    VALUE context = stream_context_alloc(stream_context_class);
    
    // Initialize it
    stream_context_initialize(context, stream_or_device);
    
    return context;
}

// Initialize stream module
void init_stream(VALUE module) {
    // Define Stream class
    VALUE stream_class = rb_define_class_under(module, "Stream", rb_cObject);
    rb_define_alloc_func(stream_class, stream_alloc);
    rb_define_method(stream_class, "initialize", RUBY_METHOD_FUNC(stream_initialize), -1);
    rb_define_method(stream_class, "synchronize", RUBY_METHOD_FUNC(stream_synchronize), 0);
    rb_define_method(stream_class, "device", RUBY_METHOD_FUNC(stream_device), 0);
    rb_define_method(stream_class, "==", RUBY_METHOD_FUNC(stream_equal), 1);
    rb_define_method(stream_class, "inspect", RUBY_METHOD_FUNC(stream_inspect), 0);
    rb_define_alias(stream_class, "to_s", "inspect");
    
    // Define StreamContext class
    VALUE stream_context_class = rb_define_class_under(module, "StreamContext", rb_cObject);
    rb_define_alloc_func(stream_context_class, stream_context_alloc);
    rb_define_method(stream_context_class, "initialize", RUBY_METHOD_FUNC(stream_context_initialize), 1);
    rb_define_method(stream_context_class, "enter", RUBY_METHOD_FUNC(stream_context_enter), 0);
    rb_define_method(stream_context_class, "exit", RUBY_METHOD_FUNC(stream_context_exit), 3);
    
    // Define module functions
    rb_define_module_function(module, "default_stream", RUBY_METHOD_FUNC(default_stream), 1);
    rb_define_module_function(module, "set_default_stream", RUBY_METHOD_FUNC(set_default_stream), 1);
    rb_define_module_function(module, "new_stream", RUBY_METHOD_FUNC(new_stream), 1);
    rb_define_module_function(module, "synchronize", RUBY_METHOD_FUNC(synchronize_module), -1);
    
    // Define stream convenience function (returns a StreamContext)
    rb_define_module_function(module, "stream", RUBY_METHOD_FUNC(stream_function), 1);
} 