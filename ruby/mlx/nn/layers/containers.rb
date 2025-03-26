module MLX
  module NN
    module Layers
      # Sequential container that passes input through modules in order
      class Sequential < MLX::NN::Module
        def initialize(*modules)
          super()
          @modules = modules
          
          # Register modules
          modules.each_with_index do |mod, i|
            register_module(i.to_s, mod)
          end
        end
        
        # Add a module to the container
        def append(mod)
          register_module(@modules.length.to_s, mod)
          @modules << mod
          self
        end
        
        # Execute modules in sequence
        def forward(x)
          @modules.each do |mod|
            x = mod.forward(x)
          end
          x
        end
        
        # Get number of modules
        def length
          @modules.length
        end
        
        # Access module by index
        def [](idx)
          @modules[idx]
        end
        
        # Iterate through modules
        def each(&block)
          @modules.each(&block)
        end
        
        # Map each module
        def map(&block)
          @modules.map(&block)
        end
        
        # Check if contains any modules
        def empty?
          @modules.empty?
        end
        
        # Get first module
        def first
          @modules.first
        end
        
        # Get last module
        def last
          @modules.last
        end
      end
    end
  end
end 