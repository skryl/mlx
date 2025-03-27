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

    modules = Core.modules + Core.classes 

    # Dynamically assign constants and accessor methods
    (Core.modules + Core.classes).each do |mod|
      const_name = mod.name.split('::').last
      accessor_name = const_name.downcase.to_sym

      # Assign module to constant under MLX::
      const_set(const_name, mod) unless const_defined?("MLX::#{const_name}")

      # Define accessor method for module (e.g., MLX.fft => MLX::FFT)
      define_singleton_method(accessor_name) do
        const_get(const_name)
      end unless respond_to?(accessor_name)
    end

    # Delegate static MLX.method calls to static module or class methods
    class << self
      def method_missing(name, *args, &block)
        target_mod = (Core.modules + Core.classes).find { |mod| mod.respond_to?(name) }
        
        if target_mod
          if args.length > 0
            # If the method returns a class and we're passing arguments, we need to instantiate it
            return target_mod.public_send(name).send(:new, *args, &block)
          else
            # If the method returns a class and we're not passing arguments, we can just call it
            return target_mod.public_send(name, &block)
          end
        end
        
        super
      end

      def respond_to_missing?(name, include_private = false)
        Core.modules.any? { |mod| mod.respond_to?(name) } || 
        Core.classes.any? { |klass| klass.respond_to?(name) } || 
        super
      end
    end

end