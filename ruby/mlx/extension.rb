module MLX
  # Extension module for adding custom operations and their derivatives
  module Extension
    # Register a new primitive operation
    # 
    # @param name [String] Name of the primitive operation
    # @param fn [Proc] Function implementation
    # @param vjp [Proc] Vector-Jacobian product function for backward pass
    # @param jvp [Proc] Jacobian-vector product function for forward pass
    # @param impl [Symbol] Implementation type (:cpu or :gpu)
    # @return [Proc] The registered function
    def self.register_primitive(name, fn, vjp: nil, jvp: nil, impl: nil)
      # Check if name is already registered
      if MLX.respond_to?(name)
        raise ArgumentError, "Function '#{name}' is already registered"
      end
      
      # Define the function in the MLX module
      MLX.define_singleton_method(name, fn)
      
      # Register VJP and JVP functions if provided
      if vjp
        vjp_map[name] = vjp
      end
      
      if jvp
        jvp_map[name] = jvp
      end
      
      # Register implementation type
      if impl
        impl_map[name] = impl
      end
      
      # Return the function
      MLX.method(name)
    end
    
    # Get the hash map of VJP functions
    # @private
    def self.vjp_map
      @vjp_map ||= {}
    end
    
    # Get the hash map of JVP functions
    # @private
    def self.jvp_map
      @jvp_map ||= {}
    end
    
    # Get the hash map of implementation types
    # @private
    def self.impl_map
      @impl_map ||= {}
    end
    
    # Register a custom derivative for a primitive operation
    # 
    # @param fn_name [String] Name of the primitive operation
    # @param vjp [Proc] Vector-Jacobian product function
    # @return [nil]
    def self.register_vjp(fn_name, vjp)
      unless MLX.respond_to?(fn_name)
        raise ArgumentError, "Function '#{fn_name}' is not registered"
      end
      
      vjp_map[fn_name] = vjp
      nil
    end
    
    # Register a custom forward-mode derivative for a primitive operation
    # 
    # @param fn_name [String] Name of the primitive operation
    # @param jvp [Proc] Jacobian-vector product function
    # @return [nil]
    def self.register_jvp(fn_name, jvp)
      unless MLX.respond_to?(fn_name)
        raise ArgumentError, "Function '#{fn_name}' is not registered"
      end
      
      jvp_map[fn_name] = jvp
      nil
    end
    
    # Custom operation class for defining new operations
    class CustomOp
      attr_reader :name, :implementation
      
      # Initialize a new custom operation
      # 
      # @param name [String] Name of the custom operation
      # @param implementation [Proc] Implementation function
      def initialize(name, implementation)
        @name = name
        @implementation = implementation
        @vjp = nil
        @jvp = nil
        @impl_type = nil
      end
      
      # Define the backward pass for this operation
      # 
      # @param vjp [Proc] Vector-Jacobian product function
      # @return [CustomOp] self
      def def_vjp(vjp)
        @vjp = vjp
        self
      end
      
      # Define the forward pass for this operation
      # 
      # @param jvp [Proc] Jacobian-vector product function
      # @return [CustomOp] self
      def def_jvp(jvp)
        @jvp = jvp
        self
      end
      
      # Define the implementation type
      # 
      # @param type [Symbol] Implementation type (:cpu or :gpu)
      # @return [CustomOp] self
      def def_impl(type)
        unless [:cpu, :gpu].include?(type)
          raise ArgumentError, "Implementation type must be :cpu or :gpu"
        end
        
        @impl_type = type
        self
      end
      
      # Register this custom operation with MLX
      # 
      # @return [Proc] The registered function
      def register
        Extension.register_primitive(
          @name, 
          @implementation, 
          vjp: @vjp,
          jvp: @jvp,
          impl: @impl_type
        )
      end
    end
    
    # Create a new custom operation
    # 
    # @param name [String] Name of the custom operation
    # @param implementation [Proc] Implementation function
    # @return [CustomOp] A new custom operation
    def self.custom_op(name, implementation)
      CustomOp.new(name, implementation)
    end
    
    # Run a function with a custom implementation for a specific device
    # 
    # @param fn [Proc] Function to run
    # @param args [Array] Arguments to pass to the function
    # @param impl [Symbol] Implementation to use (:cpu or :gpu)
    # @return [Object] Result of the function
    def self.with_impl(fn, *args, impl: :cpu)
      # Save the current implementation type
      old_impl = current_impl
      
      begin
        # Set the implementation type
        self.current_impl = impl
        
        # Call the function
        fn.call(*args)
      ensure
        # Restore the implementation type
        self.current_impl = old_impl
      end
    end
    
    # Get the current implementation type
    # 
    # @return [Symbol] Current implementation type
    def self.current_impl
      @current_impl ||= :cpu
    end
    
    # Set the current implementation type
    # 
    # @param impl [Symbol] Implementation type (:cpu or :gpu)
    # @return [Symbol] New implementation type
    def self.current_impl=(impl)
      unless [:cpu, :gpu].include?(impl)
        raise ArgumentError, "Implementation type must be :cpu or :gpu"
      end
      
      @current_impl = impl
    end
    
    # Run the CPU implementation of a function
    # 
    # @param fn [Proc] Function to run
    # @param args [Array] Arguments to pass to the function
    # @return [Object] Result of the function
    def self.cpu_impl(fn, *args)
      with_impl(fn, *args, impl: :cpu)
    end
    
    # Run the GPU implementation of a function
    # 
    # @param fn [Proc] Function to run
    # @param args [Array] Arguments to pass to the function
    # @return [Object] Result of the function
    def self.gpu_impl(fn, *args)
      with_impl(fn, *args, impl: :gpu)
    end
  end
end 