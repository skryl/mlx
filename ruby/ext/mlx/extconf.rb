require 'mkmf'

# Detect MLX installation
def find_mlx
  # First check if MLX_DIR is set in the environment
  mlx_dir = ENV['MLX_DIR']
  
  if mlx_dir
    $INCFLAGS << " -I#{mlx_dir}/include"
    $LDFLAGS << " -L#{mlx_dir}/lib -lmlx"
    return true
  end
  
  # Try to find MLX in standard locations
  ['/usr/local', '/opt/local', '/usr'].each do |prefix|
    if File.exist?("#{prefix}/include/mlx/array.h")
      $INCFLAGS << " -I#{prefix}/include"
      $LDFLAGS << " -L#{prefix}/lib -lmlx"
      return true
    end
  end
  
  # If we're on macOS, try the Homebrew location
  if RUBY_PLATFORM =~ /darwin/
    brew_prefix = `brew --prefix`.chomp rescue nil
    if brew_prefix && File.exist?("#{brew_prefix}/include/mlx/array.h")
      $INCFLAGS << " -I#{brew_prefix}/include"
      $LDFLAGS << " -L#{brew_prefix}/lib -lmlx"
      return true
    end
  end
  
  false
end

# Add C++17 flags
$CXXFLAGS << " -std=c++17"

# Find MLX
unless find_mlx
  abort "*** ERROR: MLX library not found. Please install MLX or specify MLX_DIR."
end

# Create Makefile
create_makefile('mlx/core') 