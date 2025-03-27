# MLX Ruby - Ruby bindings for Apple's MLX machine learning framework
# This is the main entry point for the MLX Ruby library.
#
# The library is structured as follows:
# - mlx/ext/mlxc - C extension for core functionality
# - mlx/core.rb - Ruby wrapper for core functionality
# - mlx/nn/* - Neural network modules and layers
# - mlx/optimizers/* - Optimizers and learning rate schedulers

require 'mlx/version'
require 'mlx/core'

# Neural network core
require 'mlx/nn/base'
require 'mlx/nn/init'
require 'mlx/nn/loss'

# Optimizers and schedulers
require 'mlx/optimizers/optimizers'
require 'mlx/optimizers/schedulers'

# Neural network layers
require 'mlx/nn/layers/linear'
require 'mlx/nn/layers/activations'
require 'mlx/nn/layers/containers'
require 'mlx/nn/layers/normalization'
require 'mlx/nn/layers/convolution'
require 'mlx/nn/layers/pooling'
require 'mlx/nn/layers/recurrent'
require 'mlx/nn/layers/transformer'
require 'mlx/nn/layers/dropout'
require 'mlx/nn/layers/embedding'
require 'mlx/nn/layers/positional_encoding'
require 'mlx/nn/layers/upsample'

module MLX
  
    def self.array(data, dtype = nil); Core::Array.new(data, dtype: dtype); end

    # Method aliases at the module level
    def self.version
        Core.version
    end
    
    # Constants module methods
    def self.pi; Core.pi; end
    def self.e; Core.e; end
    def self.euler_gamma; Core.euler_gamma; end
    def self.inf; Core.inf; end
    def self.nan; Core.nan; end
    def self.newaxis; Core.newaxis; end


end