require 'rbconfig'
require 'mkmf'
require 'fileutils'

# Ensure the extension is compiled with the C++ compiler
RbConfig::MAKEFILE_CONFIG['CC']  = RbConfig::CONFIG['CXX']
RbConfig::MAKEFILE_CONFIG['LD']  = RbConfig::CONFIG['CXX']
RbConfig::MAKEFILE_CONFIG['CXXFLAGS'] ||= ""
RbConfig::MAKEFILE_CONFIG['CXXFLAGS'] << " -std=c++17"

# Set LDSHARED to the C++ linker
RbConfig::MAKEFILE_CONFIG['LDSHARED'] = RbConfig::CONFIG['LDSHAREDXX']

# Add include paths for local headers
ruby_include_dir = File.expand_path('../../include', __dir__)
$INCFLAGS << " -I#{ruby_include_dir}" if File.directory?(ruby_include_dir)

src_dir = File.expand_path('../../src', __dir__)
$INCFLAGS << " -I#{src_dir}" if File.directory?(src_dir)

parent_dir = File.expand_path('../../..', __dir__)
$INCFLAGS << " -I#{parent_dir}"

# Set MLX library directory
mlx_lib_dir = ENV['MLX_LIB_DIR'] || File.expand_path('../../../build', __dir__)
puts "Using MLX libs from: #{mlx_lib_dir}"

# Add compiler and linker flags
$CXXFLAGS << " -std=c++17 -frtti -fexceptions -fmax-errors=0"

# Add macOS specific frameworks and linker flags
if RUBY_PLATFORM =~ /darwin/
  $CXXFLAGS << " -framework Foundation -framework Metal -framework MetalPerformanceShaders"
  # Link to MLX library
  $LDFLAGS << " -L#{mlx_lib_dir} -lmlx"
end

# Get all source files from ruby/src
src_dir = File.expand_path('../../src', __dir__)
source_files = Dir.glob(File.join(src_dir, '*.cpp'))

# Ensure the source files exist
if source_files.empty?
  puts "Warning: No source files found in #{src_dir}"
  # Create a minimal cpp file to compile
  File.open(File.join(File.dirname(__FILE__), 'minimal.cpp'), 'w') do |f|
    f.puts '#include <ruby.h>'
    f.puts 'extern "C" void Init_core() {'
    f.puts '  rb_define_global_const("MLX_VERSION", rb_str_new_cstr("0.1.0"));'
    f.puts '}'
  end
  
  source_files = [File.join(File.dirname(__FILE__), 'minimal.cpp')]
end

# **Ensure full paths to source files**
source_files.map! { |f| File.expand_path(f) }

# **Set the source files for compilation**
$srcs = source_files

# **Set the object files output directory**
$VPATH << src_dir
$objs = source_files.map { |f| File.basename(f, '.cpp') + '.o' }

# Create the Makefile
create_makefile('mlx/core')