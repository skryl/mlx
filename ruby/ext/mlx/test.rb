#!/usr/bin/env ruby

# Add the lib directory to the load path
$LOAD_PATH.unshift(File.expand_path('lib', __dir__))

# Load the MLX extension
require 'mlx/core'

# Print the version
puts "MLX version: #{MLX::Core.version}"

# Test the Array class
array = MLX::Core::Array.new
puts "Array: #{array}" 