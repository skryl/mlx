#include <ruby.h>
#include "mlx/device.h"

namespace mx = mlx::core;

// Device methods
VALUE rb_get_default_device(VALUE self) {
  mx::Device device = mx::get_default_device();
  return INT2NUM(static_cast<int>(device.type));
}

VALUE rb_set_default_device(VALUE self, VALUE device_type) {
  mx::Device::DeviceType type = static_cast<mx::Device::DeviceType>(NUM2INT(device_type));
  mx::Device device(type);
  mx::set_default_device(device);
  return Qnil;
}

VALUE rb_sync_device(VALUE self) {
  mx::sync_device();
  return Qnil;
}

VALUE rb_devices(VALUE self) {
  VALUE result = rb_ary_new();
  auto devices = mx::devices();
  for (const auto& device : devices) {
    rb_ary_push(result, INT2NUM(static_cast<int>(device.type)));
  }
  return result;
}

// Initialize device module
void init_device(VALUE module) {
  // Define device type constants
  rb_define_const(module, "CPU", INT2NUM(static_cast<int>(mx::Device::DeviceType::cpu)));
  rb_define_const(module, "GPU", INT2NUM(static_cast<int>(mx::Device::DeviceType::gpu)));
  
  // Define module functions
  rb_define_module_function(module, "get_default_device", RUBY_METHOD_FUNC(rb_get_default_device), 0);
  rb_define_module_function(module, "set_default_device", RUBY_METHOD_FUNC(rb_set_default_device), 1);
  rb_define_module_function(module, "sync_device", RUBY_METHOD_FUNC(rb_sync_device), 0);
  rb_define_module_function(module, "devices", RUBY_METHOD_FUNC(rb_devices), 0);
} 