#include <ruby.h>
#include <sstream>
#include "mlx/device.h"

namespace mx = mlx::core;

// Device class representation in Ruby
static void rb_free_device(void* ptr) {
  mx::Device* device = static_cast<mx::Device*>(ptr);
  delete device;
}

static VALUE rb_alloc_device(VALUE klass) {
  mx::Device* ptr = new mx::Device(mx::Device::DeviceType::cpu);
  return Data_Wrap_Struct(klass, 0, rb_free_device, ptr);
}

static VALUE rb_device_initialize(int argc, VALUE* argv, VALUE self) {
  mx::Device* device_ptr;
  Data_Get_Struct(self, mx::Device, device_ptr);
  
  VALUE type_val;
  VALUE index_val = INT2NUM(0);
  
  rb_scan_args(argc, argv, "11", &type_val, &index_val);
  
  mx::Device::DeviceType type = static_cast<mx::Device::DeviceType>(NUM2INT(type_val));
  int index = NUM2INT(index_val);
  
  *device_ptr = mx::Device(type, index);
  
  return self;
}

static VALUE rb_device_type(VALUE self) {
  mx::Device* device_ptr;
  Data_Get_Struct(self, mx::Device, device_ptr);
  return INT2NUM(static_cast<int>(device_ptr->type));
}

static VALUE rb_device_to_s(VALUE self) {
  mx::Device* device_ptr;
  Data_Get_Struct(self, mx::Device, device_ptr);
  std::ostringstream os;
  // Manually format the device output since there's no << operator
  os << "Device(type=" << (device_ptr->type == mx::Device::DeviceType::cpu ? "cpu" : "gpu") 
     << ", index=" << device_ptr->index << ")";
  return rb_str_new_cstr(os.str().c_str());
}

static VALUE rb_device_equal(VALUE self, VALUE other) {
  if (!rb_obj_is_kind_of(other, rb_class_of(self))) {
    return Qfalse;
  }
  
  mx::Device* self_device;
  mx::Device* other_device;
  
  Data_Get_Struct(self, mx::Device, self_device);
  Data_Get_Struct(other, mx::Device, other_device);
  
  return (*self_device == *other_device) ? Qtrue : Qfalse;
}

// Device module methods
VALUE rb_default_device(VALUE self) {
  mx::Device device = mx::default_device();
  VALUE device_class = rb_path2class("MLX::Device");
  mx::Device* new_device = new mx::Device(device);
  return Data_Wrap_Struct(device_class, 0, rb_free_device, new_device);
}

VALUE rb_set_default_device(VALUE self, VALUE device_obj) {
  if (rb_obj_is_kind_of(device_obj, rb_path2class("MLX::Device"))) {
    mx::Device* device_ptr;
    Data_Get_Struct(device_obj, mx::Device, device_ptr);
    mx::set_default_device(*device_ptr);
  } else {
    mx::Device::DeviceType type = static_cast<mx::Device::DeviceType>(NUM2INT(device_obj));
    mx::Device device(type);
    mx::set_default_device(device);
  }
  return Qnil;
}

VALUE rb_sync_device(VALUE self) {
  // MLX doesn't have a sync_device function directly, so we'll use
  // the appropriate alternative (no-op for now)
  return Qnil;
}

VALUE rb_devices(VALUE self) {
  VALUE result = rb_ary_new();
  VALUE device_class = rb_path2class("MLX::Device");
  
  // Create CPU device (index 0)
  mx::Device* cpu_device = new mx::Device(mx::Device::DeviceType::cpu, 0);
  rb_ary_push(result, Data_Wrap_Struct(device_class, 0, rb_free_device, cpu_device));
  
  // If GPU is available, also add it
  if (mx::default_device().type == mx::Device::DeviceType::gpu) {
    mx::Device* gpu_device = new mx::Device(mx::Device::DeviceType::gpu, 0);
    rb_ary_push(result, Data_Wrap_Struct(device_class, 0, rb_free_device, gpu_device));
  }
  
  return result;
}

// Initialize device module
void init_device(VALUE module) {
  // Define Device class
  VALUE device_class = rb_define_class_under(module, "Device", rb_cObject);
  rb_define_alloc_func(device_class, rb_alloc_device);
  
  // Device instance methods
  rb_define_method(device_class, "initialize", RUBY_METHOD_FUNC(rb_device_initialize), -1);
  rb_define_method(device_class, "type", RUBY_METHOD_FUNC(rb_device_type), 0);
  rb_define_method(device_class, "to_s", RUBY_METHOD_FUNC(rb_device_to_s), 0);
  rb_define_method(device_class, "==", RUBY_METHOD_FUNC(rb_device_equal), 1);
  
  // Define device type constants
  rb_define_const(module, "CPU", INT2NUM(static_cast<int>(mx::Device::DeviceType::cpu)));
  rb_define_const(module, "GPU", INT2NUM(static_cast<int>(mx::Device::DeviceType::gpu)));
  
  // Define module functions
  rb_define_module_function(module, "default_device", RUBY_METHOD_FUNC(rb_default_device), 0);
  rb_define_module_function(module, "set_default_device", RUBY_METHOD_FUNC(rb_set_default_device), 1);
  rb_define_module_function(module, "sync_device", RUBY_METHOD_FUNC(rb_sync_device), 0);
  rb_define_module_function(module, "devices", RUBY_METHOD_FUNC(rb_devices), 0);
} 