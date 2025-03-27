#!/usr/bin/env ruby

# Set up the load paths for the MLX gem
ruby_dir = File.expand_path('..', __dir__)
lib_dir = File.join(ruby_dir, 'lib')
$LOAD_PATH.unshift(lib_dir) unless $LOAD_PATH.include?(lib_dir)
$LOAD_PATH.unshift(ruby_dir) unless $LOAD_PATH.include?(ruby_dir)

# Point to the build directory for the MLX library
build_dir = File.expand_path('../../build', __dir__)
ENV['DYLD_LIBRARY_PATH'] = "#{build_dir}:#{ENV['DYLD_LIBRARY_PATH']}"

puts "MLX Ruby Bindings Test"
puts "====================="
puts "Build directory: #{build_dir}"
puts "Ruby directory: #{ruby_dir}"
puts "Lib directory: #{lib_dir}"
puts "Load paths: #{$LOAD_PATH.join(', ')}"
puts "Current directory: #{File.dirname(__FILE__)}"

# Load version first
begin
  require 'mlx/version'
  puts "\nMLX Gem Info"
  puts "------------"
  puts "MLX version: #{MLX::VERSION}"
rescue LoadError => e
  puts "Error loading version: #{e.message}"
end

# Now load the core module
begin
  require 'mlx/core'
  puts "MLX Core loaded successfully"
rescue LoadError => e
  puts "Error loading core: #{e.message}"
end

# Now try to load the main mlx file
begin
  require 'mlx'
  puts "MLX main module loaded successfully"
rescue LoadError => e
  puts "Error loading main module: #{e.message}"
end

# Test array creation
puts "\nTesting Array Class"
puts "----------------"
begin
  array = MLX::Core::Array.new
  puts "Created array: #{array}"
rescue => e
  puts "Error creating array: #{e.message}"
end

# Test other MLX modules
puts "\nTesting Other Modules"
puts "-------------------"

# List of modules to test
test_modules = [
  ['MLX::Core::Device', :default],
  ['MLX::Core::Random', :key],
  ['MLX::Core::Metal', :available?],
  ['MLX::Core::Constants', :PI],
  ['MLX::NN', :version],
  ['MLX::Optimizers', :version],
  ['MLX::Fast', :version],
  ['MLX::Utils', :version]
]

test_modules.each do |module_name, method_or_const|
  begin
    module_parts = module_name.split('::')
    mod = Object
    module_parts.each { |part| mod = mod.const_get(part) }
    
    if method_or_const.is_a?(Symbol) && mod.respond_to?(method_or_const)
      result = mod.send(method_or_const)
      puts "#{module_name}.#{method_or_const}: #{result}"
    elsif mod.const_defined?(method_or_const)
      result = mod.const_get(method_or_const)
      puts "#{module_name}::#{method_or_const}: #{result}"
    else
      puts "#{module_name}: Available"
    end
  rescue => e
    puts "#{module_name}: Error - #{e.message}"
  end
end

# List all available MLX components
puts "\nAvailable MLX Components:"
puts "----------------------"

def list_module(mod, prefix = '')
  mod.constants.sort.each do |const|
    begin
      value = mod.const_get(const)
      name = "#{prefix}#{const}"
      
      if value.is_a?(Module) && !value.is_a?(Class)
        puts "Module: #{name}"
        list_module(value, "#{name}::")
      elsif value.is_a?(Class)
        puts "Class: #{name}"
      end
    rescue => e
      # Skip any errors when accessing constants
    end
  end
end

list_module(MLX)

puts "\nTest completed!"