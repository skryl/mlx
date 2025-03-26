module MLX
  module NN
    # Base class for all neural network modules
    class Module
      attr_reader :training

      def initialize
        @training = true
        @_parameters = {}
        @_submodules = {}
      end

      # Toggle training mode
      def train(mode = true)
        @training = mode
        submodules.each { |_, m| m.train(mode) }
        self
      end

      # Toggle eval mode
      def eval
        train(false)
      end

      # Get all trainable parameters recursively
      def trainable_parameters
        params = {}
        
        parameters.each do |name, param|
          params[name] = param
        end
        
        submodules.each do |name, mod|
          mod_params = mod.trainable_parameters
          mod_params.each do |param_name, param|
            params["#{name}.#{param_name}"] = param
          end
        end
        
        params
      end

      # Get all parameters recursively
      def parameters
        @_parameters.dup
      end

      # Get all submodules
      def submodules
        @_submodules.dup
      end

      # Update parameters with new values
      def update(params)
        params.each do |name, value|
          if name.include?(".")
            parts = name.split(".", 2)
            module_name = parts[0]
            rest = parts[1]
            
            if @_submodules.key?(module_name)
              @_submodules[module_name].update({rest => value})
            end
          else
            if @_parameters.key?(name)
              @_parameters[name] = value
            end
          end
        end

        self
      end

      # Forward pass - to be implemented by subclasses
      def forward(x)
        raise NotImplementedError, "Subclass must implement abstract method"
      end
      alias_method :call, :forward

      # Register a parameter
      def register_parameter(name, param)
        @_parameters[name] = param
      end

      # Register a submodule
      def register_module(name, module_obj)
        @_submodules[name] = module_obj
      end

      # Reset parameters
      def reset_parameters
        # To be implemented by subclasses
      end

      # State dict for serialization
      def state_dict
        state = {}
        
        parameters.each do |name, param|
          state[name] = param
        end
        
        submodules.each do |name, mod|
          mod_state = mod.state_dict
          mod_state.each do |key, value|
            state["#{name}.#{key}"] = value
          end
        end
        
        state
      end

      # Load state dict
      def load_state_dict(state_dict)
        update(state_dict)
      end
    end
  end
end 