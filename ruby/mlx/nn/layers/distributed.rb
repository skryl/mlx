module MLX
  module NN
    module Layers
      # Module for distributed layer functionality
      
      # Cache for gradient summation functions
      @@sum_gradients_cache = {}
      
      # Get a function that sums gradients across a distributed group
      #
      # @param group [MLX::Distributed::Group] Group to sum gradients across
      # @return [Proc] Function that sums gradients
      def self.sum_gradients(group)
        # Return cached function if available
        return @@sum_gradients_cache[group.object_id] if @@sum_gradients_cache[group.object_id]
        
        # If group has size 1, just return identity function
        if group.size == 1
          fn = ->(x) { x }
          @@sum_gradients_cache[group.object_id] = fn
          return fn
        end
        
        # Create a custom function with VJP for gradient summing
        fn = MLX::Extension.custom_op("sum_gradients", ->(x) { x })
                           .def_vjp(->(x, dx, _) { MLX::Distributed.all_sum(dx, group: group) })
                           .register
        
        # Cache and return the function
        @@sum_gradients_cache[group.object_id] = fn
        fn
      end
      
      # Split a weight tensor along an axis
      #
      # @param weight [MLX::Array] Weight tensor to split
      # @param segments [Integer, Array] Number of segments or segment ratios
      # @param axis [Integer] Axis to split along
      # @return [Array<MLX::Array>] Array of split tensors
      def self.split_weights(weight, segments, axis)
        # Handle integer segments (equal splits)
        if segments.is_a?(Integer) || (segments.is_a?(Array) && segments.all? { |s| s.is_a?(Integer) })
          return MLX.split(weight, segments, axis: axis)
        end
        
        # Handle fractional segments
        n = weight.shape[axis]
        indices = segments.map { |s| (s * n).to_i }
        MLX.split(weight, indices, axis: axis)
      end
      
      # Shard parameters according to a predicate function
      #
      # @param parameters [Hash] Parameter dictionary
      # @param sharding_predicate [Proc] Function that returns sharding info (axis, segments)
      # @param group [MLX::Distributed::Group, nil] Distributed group
      # @return [Hash] New parameter dictionary with sharded weights
      def self.shard_parameters(parameters, sharding_predicate, group = nil)
        # Get the distributed group and rank
        group ||= MLX::Distributed.init
        n = group.size
        r = group.rank
        
        # Function to apply to each parameter
        shard_fn = ->(path, weight) do
          # Skip non-array parameters
          return weight unless weight.is_a?(MLX::Array)
          
          # Get sharding info from predicate
          sharding_info = sharding_predicate.call(path, weight)
          return weight if sharding_info.nil?
          
          # Extract axis and segments
          if sharding_info.is_a?(Integer)
            axis = sharding_info
            segments = 1
          elsif sharding_info.is_a?(Array) || sharding_info.is_a?(Tuple)
            axis, segments = sharding_info
          else
            raise ArgumentError, "Sharding function should return Integer or [Integer, Integer/Array]"
          end
          
          # Perform the sharding
          parts = []
          split_weights(weight, segments, axis).each do |part|
            parts << split_weights(part, n, axis)[r]
          end
          
          # Concatenate and ensure contiguous memory layout
          MLX.contiguous(MLX.concatenate(parts, axis: axis))
        end
        
        # Apply sharding to all parameters
        MLX::Utils.tree_map_with_path(shard_fn, parameters)
      end
      
      # Predicate for all-to-sharded transformation
      #
      # @param segments [Integer, Array] Number of segments or segment ratios
      # @return [Proc] Sharding predicate
      def self.all_to_sharded_predicate(segments)
        ->(path, weight) { [weight.ndim > 1 ? [weight.ndim - 2, 0].max : 0, segments] }
      end
      
      # Predicate for sharded-to-all transformation
      #
      # @param segments [Integer, Array] Number of segments or segment ratios
      # @return [Proc] Sharding predicate
      def self.sharded_to_all_predicate(segments)
        ->(path, weight) { path.end_with?("bias") ? nil : [-1, segments] }
      end
      
      # Validate sharding type
      #
      # @param sharding [String] Sharding type
      # @raise [ArgumentError] If sharding type is invalid
      # @return [nil]
      def self.validate_sharding(sharding)
        unless ["all-to-sharded", "sharded-to-all"].include?(sharding)
          raise ArgumentError, "Sharding type must be 'all-to-sharded' or 'sharded-to-all'"
        end
        nil
      end
      
      # Base class for sharded linear layers
      class ShardedLinear < MLX::NN::Module
        attr_reader :in_features, :out_features, :bias, :shard_count, :shard_index
        
        def initialize(in_features, out_features, bias: true, shard_count: 1, shard_index: 0)
          super()
          
          # Validate shard parameters
          if shard_count < 1
            raise ArgumentError, "shard_count must be at least 1, got #{shard_count}"
          end
          
          if shard_index < 0 || shard_index >= shard_count
            raise ArgumentError, "shard_index must be in range [0, #{shard_count - 1}], got #{shard_index}"
          end
          
          @in_features = in_features
          @out_features = out_features
          @bias = bias
          @shard_count = shard_count
          @shard_index = shard_index
        end
      end
      
      # Linear layer that shards its weights over the output dimension
      class ShardedToAllLinear < ShardedLinear
        def initialize(in_features, out_features, bias: true, shard_count: 1, shard_index: 0)
          super(in_features, out_features, bias: bias, shard_count: shard_count, shard_index: shard_index)
          
          # Each shard handles out_features / shard_count outputs
          @shard_out_features = out_features // shard_count
          
          # Calculate the offset for this shard
          @shard_out_offset = shard_index * @shard_out_features
          
          # Create weight matrix for this shard's portion of outputs
          scale = 1.0 / Math.sqrt(in_features)
          weight = MLX::Random.uniform(-scale, scale, [@shard_out_features, in_features])
          register_parameter("weight", weight)
          
          # Create bias for this shard's portion of outputs (if needed)
          if bias
            b = MLX::Random.uniform(-scale, scale, [@shard_out_features])
            register_parameter("bias", b)
          end
        end
        
        def forward(x)
          # Compute this shard's portion of the output
          output = MLX.matmul(x, @_parameters["weight"].transpose)
          
          # Add bias if present
          output = MLX.add(output, @_parameters["bias"]) if @bias
          
          # Gather outputs from all shards
          if @shard_count > 1
            # This would use MLX.distributed.all_gather in practice
            # For now, we'll simulate it with a placeholder that just returns this shard
            output = MLX::Distributed.all_gather(output)
          end
          
          output
        end
      end
      
      # Linear layer that takes sharded inputs and gathers them
      class AllToShardedLinear < ShardedLinear
        def initialize(in_features, out_features, bias: true, shard_count: 1, shard_index: 0)
          super(in_features, out_features, bias: bias, shard_count: shard_count, shard_index: shard_index)
          
          # Each shard processes in_features / shard_count inputs
          @shard_in_features = in_features // shard_count
          
          # Calculate the offset for this shard
          @shard_in_offset = shard_index * @shard_in_features
          
          # Create weight matrix for this shard's portion of inputs
          scale = 1.0 / Math.sqrt(@shard_in_features)
          weight = MLX::Random.uniform(-scale, scale, [out_features, @shard_in_features])
          register_parameter("weight", weight)
          
          # Create bias (only on the first shard to avoid duplication)
          if bias && shard_index == 0
            b = MLX::Random.uniform(-scale, scale, [out_features])
            register_parameter("bias", b)
          end
        end
        
        def forward(x)
          # Each shard computes with its portion of the input
          shard_output = MLX.matmul(x, @_parameters["weight"].transpose)
          
          # Sum outputs across all shards
          if @shard_count > 1
            # This would use MLX.distributed.all_reduce with 'sum' in practice
            # For now, we'll simulate it with a placeholder
            output = MLX::Distributed.all_reduce(shard_output, 'sum')
          else
            output = shard_output
          end
          
          # Add bias if present (only on first shard)
          if @bias && @shard_index == 0
            output = MLX.add(output, @_parameters["bias"])
          end
          
          output
        end
      end
      
      # Quantized version of ShardedToAllLinear
      class QuantizedShardedToAllLinear < ShardedLinear
        attr_reader :group_size, :bits
        
        def initialize(in_features, out_features, bias: true, shard_count: 1, shard_index: 0, 
                       group_size: 64, bits: 4)
          super(in_features, out_features, bias: bias, shard_count: shard_count, shard_index: shard_index)
          
          # Quantization parameters
          @group_size = group_size
          @bits = bits
          
          # Each shard handles out_features / shard_count outputs
          @shard_out_features = out_features // shard_count
          
          # Calculate the offset for this shard
          @shard_out_offset = shard_index * @shard_out_features
          
          # Initialize quantized parameters
          qweight = MLX.zeros([@shard_out_features, in_features], dtype: MLX::INT8)
          qzeros = MLX.zeros([@shard_out_features, (in_features // group_size) * (bits // 8)], dtype: MLX::INT8)
          scales = MLX.ones([@shard_out_features, in_features // group_size], dtype: MLX::FLOAT32)
          
          register_parameter("qweight", qweight)
          register_parameter("qzeros", qzeros)
          register_parameter("scales", scales)
          
          if bias
            b = MLX.zeros([@shard_out_features], dtype: MLX::FLOAT32)
            register_parameter("bias", b)
          end
        end
        
        # Create from an existing ShardedToAllLinear layer
        def self.from_linear(linear, weight_params: {}, **kwargs)
          # Extract parameters
          in_features = linear.in_features
          out_features = linear.out_features
          has_bias = linear.bias
          shard_count = linear.shard_count
          shard_index = linear.shard_index
          
          # Default quantization parameters
          group_size = weight_params[:group_size] || 64
          bits = weight_params[:bits] || 4
          
          # Create quantized version
          quantized = QuantizedShardedToAllLinear.new(
            in_features, 
            out_features, 
            bias: has_bias,
            shard_count: shard_count,
            shard_index: shard_index,
            group_size: group_size,
            bits: bits
          )
          
          # Quantize weights (simplified implementation)
          # Real implementation would use proper quantization algorithms
          weight = linear.weight
          shard_out_features = weight.shape[0]
          
          # Group the weights
          groups = in_features // group_size
          reshaped_weight = weight.reshape([shard_out_features, groups, group_size])
          
          # Calculate scales (max value in each group)
          scales = MLX.max(MLX.abs(reshaped_weight), axis: 2) / ((2 ** (bits - 1)) - 1)
          # Avoid division by zero
          scales = MLX.maximum(scales, 1e-9)
          
          # Quantize weights
          qweight_float = MLX.clip(
            MLX.round(reshaped_weight / scales.reshape([shard_out_features, groups, 1])),
            -(2 ** (bits - 1)),
            (2 ** (bits - 1)) - 1
          )
          
          # Convert to integers
          qweight = qweight_float.astype(MLX::INT8).reshape([shard_out_features, in_features])
          
          # Set parameters
          quantized.instance_variable_set("@_parameters", {
            "qweight" => qweight,
            "qzeros" => MLX.zeros([shard_out_features, (in_features // group_size) * (bits // 8)], dtype: MLX::INT8),
            "scales" => scales
          })
          
          # Set bias if present
          if has_bias
            quantized.instance_variable_set("@_parameters", 
              quantized.instance_variable_get("@_parameters").merge({"bias" => linear.bias})
            )
          end
          
          quantized
        end
        
        def forward(x)
          # Dequantize weights for the forward pass
          weight = dequantize_weight
          
          # Compute this shard's portion of the output
          output = MLX.matmul(x, weight.transpose)
          
          # Add bias if present
          output = MLX.add(output, @_parameters["bias"]) if @bias
          
          # Gather outputs from all shards
          if @shard_count > 1
            # This would use MLX.distributed.all_gather in practice
            output = MLX::Distributed.all_gather(output)
          end
          
          output
        end
        
        # Dequantize the weights for computation
        def dequantize_weight
          qweight = @_parameters["qweight"]
          scales = @_parameters["scales"]
          
          # Reshape for broadcasting
          qweight_reshaped = qweight.reshape([@shard_out_features, @in_features // @group_size, @group_size])
          scales_reshaped = scales.reshape([@shard_out_features, @in_features // @group_size, 1])
          
          # Dequantize
          dequantized = qweight_reshaped.astype(MLX::FLOAT32) * scales_reshaped
          
          # Reshape back to original shape
          dequantized.reshape([@shard_out_features, @in_features])
        end
      end
      
      # Quantized version of AllToShardedLinear
      class QuantizedAllToShardedLinear < ShardedLinear
        attr_reader :group_size, :bits
        
        def initialize(in_features, out_features, bias: true, shard_count: 1, shard_index: 0, 
                       group_size: 64, bits: 4)
          super(in_features, out_features, bias: bias, shard_count: shard_count, shard_index: shard_index)
          
          # Quantization parameters
          @group_size = group_size
          @bits = bits
          
          # Each shard processes in_features / shard_count inputs
          @shard_in_features = in_features // shard_count
          
          # Initialize quantized parameters
          qweight = MLX.zeros([out_features, @shard_in_features], dtype: MLX::INT8)
          qzeros = MLX.zeros([out_features, (@shard_in_features // group_size) * (bits // 8)], dtype: MLX::INT8)
          scales = MLX.ones([out_features, @shard_in_features // group_size], dtype: MLX::FLOAT32)
          
          register_parameter("qweight", qweight)
          register_parameter("qzeros", qzeros)
          register_parameter("scales", scales)
          
          # Create bias (only on the first shard to avoid duplication)
          if bias && shard_index == 0
            b = MLX.zeros([out_features], dtype: MLX::FLOAT32)
            register_parameter("bias", b)
          end
        end
        
        # Create from an existing AllToShardedLinear layer
        def self.from_linear(linear, weight_params: {}, **kwargs)
          # Extract parameters
          in_features = linear.in_features
          out_features = linear.out_features
          has_bias = linear.bias
          shard_count = linear.shard_count
          shard_index = linear.shard_index
          
          # Default quantization parameters
          group_size = weight_params[:group_size] || 64
          bits = weight_params[:bits] || 4
          
          # Create quantized version
          quantized = QuantizedAllToShardedLinear.new(
            in_features, 
            out_features, 
            bias: has_bias,
            shard_count: shard_count,
            shard_index: shard_index,
            group_size: group_size,
            bits: bits
          )
          
          # Quantize weights (simplified implementation)
          weight = linear.weight
          shard_in_features = weight.shape[1]
          
          # Group the weights
          groups = shard_in_features // group_size
          reshaped_weight = weight.reshape([out_features, groups, group_size])
          
          # Calculate scales (max value in each group)
          scales = MLX.max(MLX.abs(reshaped_weight), axis: 2) / ((2 ** (bits - 1)) - 1)
          # Avoid division by zero
          scales = MLX.maximum(scales, 1e-9)
          
          # Quantize weights
          qweight_float = MLX.clip(
            MLX.round(reshaped_weight / scales.reshape([out_features, groups, 1])),
            -(2 ** (bits - 1)),
            (2 ** (bits - 1)) - 1
          )
          
          # Convert to integers
          qweight = qweight_float.astype(MLX::INT8).reshape([out_features, shard_in_features])
          
          # Set parameters
          quantized.instance_variable_set("@_parameters", {
            "qweight" => qweight,
            "qzeros" => MLX.zeros([out_features, (shard_in_features // group_size) * (bits // 8)], dtype: MLX::INT8),
            "scales" => scales
          })
          
          # Set bias if present (only on first shard)
          if has_bias && shard_index == 0
            quantized.instance_variable_set("@_parameters", 
              quantized.instance_variable_get("@_parameters").merge({"bias" => linear.bias})
            )
          end
          
          quantized
        end
        
        def forward(x)
          # Dequantize weights for the forward pass
          weight = dequantize_weight
          
          # Each shard computes with its portion of the input
          shard_output = MLX.matmul(x, weight.transpose)
          
          # Sum outputs across all shards
          if @shard_count > 1
            # This would use MLX.distributed.all_reduce with 'sum' in practice
            output = MLX::Distributed.all_reduce(shard_output, 'sum')
          else
            output = shard_output
          end
          
          # Add bias if present (only on first shard)
          if @bias && @shard_index == 0
            output = MLX.add(output, @_parameters["bias"])
          end
          
          output
        end
        
        # Dequantize the weights for computation
        def dequantize_weight
          qweight = @_parameters["qweight"]
          scales = @_parameters["scales"]
          
          # Reshape for broadcasting
          qweight_reshaped = qweight.reshape([@out_features, @shard_in_features // @group_size, @group_size])
          scales_reshaped = scales.reshape([@out_features, @shard_in_features // @group_size, 1])
          
          # Dequantize
          dequantized = qweight_reshaped.astype(MLX::FLOAT32) * scales_reshaped
          
          # Reshape back to original shape
          dequantized.reshape([@out_features, @shard_in_features])
        end
      end
      
      # Shard a module in-place by updating its parameter dictionary with
      # sharded parameter dictionary
      #
      # @param module_instance [MLX::NN::Module] Module to shard
      # @param sharding [String, Proc] Sharding type ('all-to-sharded', 'sharded-to-all') or custom function
      # @param segments [Integer, Array] Number of segments or segment ratios
      # @param group [MLX::Distributed::Group, nil] Distributed group
      # @return [MLX::NN::Module] The sharded module
      def self.shard_inplace(module_instance, sharding, segments: 1, group: nil)
        # Get sharding predicate based on type
        if sharding.is_a?(String)
          validate_sharding(sharding)
          
          predicate = if sharding == "all-to-sharded"
            all_to_sharded_predicate(segments)
          else
            sharded_to_all_predicate(segments)
          end
        else
          # Custom sharding function
          predicate = sharding
        end
        
        # Apply sharding to module parameters
        module_instance.update(shard_parameters(module_instance.parameters, predicate, group))
        module_instance
      end
      
      # Create a new distributed linear layer with appropriate sharding
      #
      # @param module_instance [MLX::NN::Layers::Linear] Linear layer to shard
      # @param sharding [String] Sharding type ('all-to-sharded', 'sharded-to-all')
      # @param segments [Integer, Array] Number of segments or segment ratios
      # @param group [MLX::Distributed::Group, nil] Distributed group
      # @return [MLX::NN::Module] A new sharded linear layer
      def self.shard_linear(module_instance, sharding, segments: 1, group: nil)
        validate_sharding(sharding)
        
        # Determine the appropriate factory method based on module type and sharding type
        is_quantized = module_instance.is_a?(MLX::NN::Layers::QuantizedLinear)
        
        factory_methods = {
          ["all-to-sharded", false] => AllToShardedLinear.method(:from_linear),
          ["all-to-sharded", true] => QuantizedAllToShardedLinear.method(:from_quantized_linear),
          ["sharded-to-all", false] => ShardedToAllLinear.method(:from_linear),
          ["sharded-to-all", true] => QuantizedShardedToAllLinear.method(:from_quantized_linear)
        }
        
        # Create the sharded layer
        factory_methods[[sharding, is_quantized]].call(
          module_instance, 
          segments: segments, 
          group: group
        )
      end
      
      # Module-level functions for creating distributed layers

      # Create an all-to-sharded linear layer from an existing linear layer
      # 
      # @param linear [MLX::NN::Layers::Linear] Original linear layer
      # @param segments [Integer, Array] Number of segments or segment ratios
      # @param group [MLX::Distributed::Group, nil] Distributed group
      # @return [AllToShardedLinear] A new all-to-sharded linear layer
      def self.all_to_sharded_linear(linear, segments: 1, group: nil)
        shard_linear(linear, "all-to-sharded", segments: segments, group: group)
      end
      
      # Create a sharded-to-all linear layer from an existing linear layer
      # 
      # @param linear [MLX::NN::Layers::Linear] Original linear layer
      # @param segments [Integer, Array] Number of segments or segment ratios
      # @param group [MLX::Distributed::Group, nil] Distributed group
      # @return [ShardedToAllLinear] A new sharded-to-all linear layer
      def self.sharded_to_all_linear(linear, segments: 1, group: nil)
        shard_linear(linear, "sharded-to-all", segments: segments, group: group)
      end
    end
  end
end 