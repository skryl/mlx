#include <ruby.h>
#include "mlx/stream.h"

namespace mx = mlx::core;

// Stream class
typedef struct {
    mx::Stream* stream;
} StreamWrapper;

// GC callback for Stream
static void stream_free(void* ptr) {
    StreamWrapper* wrapper = static_cast<StreamWrapper*>(ptr);
    delete wrapper->stream;
    ruby_xfree(wrapper);
}

// Allocate a new Stream
static VALUE stream_alloc(VALUE klass) {
    StreamWrapper* wrapper = ALLOC(StreamWrapper);
    wrapper->stream = new mx::Stream();
    return Data_Wrap_Struct(klass, 0, stream_free, wrapper);
}

// Initialize a Stream
static VALUE stream_initialize(VALUE self) {
    return self;
}

// Get the Stream
static mx::Stream& get_stream(VALUE self) {
    StreamWrapper* wrapper;
    Data_Get_Struct(self, StreamWrapper, wrapper);
    return *(wrapper->stream);
}

// Stream methods
static VALUE stream_synchronize(VALUE self) {
    mx::Stream& stream = get_stream(self);
    stream.synchronize();
    return Qnil;
}

// Get the default stream
static VALUE get_current_stream(VALUE self) {
    mx::Stream stream = mx::get_current_stream();
    StreamWrapper* wrapper = ALLOC(StreamWrapper);
    wrapper->stream = new mx::Stream(stream);
    return Data_Wrap_Struct(rb_class_of(self), 0, stream_free, wrapper);
}

// Initialize stream module
void init_stream(VALUE module) {
    // Define Stream class
    VALUE stream_class = rb_define_class_under(module, "Stream", rb_cObject);
    rb_define_alloc_func(stream_class, stream_alloc);
    rb_define_method(stream_class, "initialize", RUBY_METHOD_FUNC(stream_initialize), 0);
    rb_define_method(stream_class, "synchronize", RUBY_METHOD_FUNC(stream_synchronize), 0);
    
    // Define module functions
    rb_define_module_function(module, "get_current_stream", RUBY_METHOD_FUNC(get_current_stream), 0);
} 