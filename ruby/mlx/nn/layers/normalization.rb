module MLX
  module NN
    module Layers
      # Layer Normalization
      class LayerNorm < MLX::NN::Module
        attr_reader :normalized_shape, :eps, :elementwise_affine
        
        def initialize(normalized_shape, eps: 1e-5, elementwise_affine: true)
          super()
          
          @normalized_shape = normalized_shape.is_a?(Array) ? normalized_shape : [normalized_shape]
          @eps = eps
          @elementwise_affine = elementwise_affine
          
          if elementwise_affine
            weight = MLX.ones(@normalized_shape)
            bias = MLX.zeros(@normalized_shape)
            
            register_parameter('weight', weight)
            register_parameter('bias', bias)
          end
        end
        
        def forward(x)
          # Determine the dimensions to normalize over
          # (all dimensions except the batch dimension)
          norm_dims = (x.ndim - @normalized_shape.length).upto(x.ndim - 1).to_a
          
          # Calculate the mean and variance
          mean = MLX.mean(x, axis: norm_dims, keepdim: true)
          var = MLX.mean(MLX.power(MLX.subtract(x, mean), 2), axis: norm_dims, keepdim: true)
          
          # Normalize
          normalized = MLX.divide(MLX.subtract(x, mean), MLX.sqrt(MLX.add(var, @eps)))
          
          if @elementwise_affine
            # Apply scale and bias
            normalized = MLX.add(MLX.multiply(normalized, @_parameters['weight']), @_parameters['bias'])
          end
          
          normalized
        end
        
        def reset_parameters
          if @elementwise_affine
            @_parameters['weight'] = MLX.ones(@normalized_shape)
            @_parameters['bias'] = MLX.zeros(@normalized_shape)
          end
        end
      end
      
      # RMS (Root Mean Square) Normalization
      class RMSNorm < MLX::NN::Module
        attr_reader :normalized_shape, :eps, :elementwise_affine
        
        def initialize(normalized_shape, eps: 1e-5, elementwise_affine: true)
          super()
          
          @normalized_shape = normalized_shape.is_a?(Array) ? normalized_shape : [normalized_shape]
          @eps = eps
          @elementwise_affine = elementwise_affine
          
          if elementwise_affine
            weight = MLX.ones(@normalized_shape)
            register_parameter('weight', weight)
          end
        end
        
        def forward(x)
          # Determine the dimensions to normalize over
          norm_dims = (x.ndim - @normalized_shape.length).upto(x.ndim - 1).to_a
          
          # Calculate the RMS
          var = MLX.mean(MLX.power(x, 2), axis: norm_dims, keepdim: true)
          rms = MLX.sqrt(MLX.add(var, @eps))
          
          # Normalize
          normalized = MLX.divide(x, rms)
          
          if @elementwise_affine
            # Apply scale
            normalized = MLX.multiply(normalized, @_parameters['weight'])
          end
          
          normalized
        end
        
        def reset_parameters
          if @elementwise_affine
            @_parameters['weight'] = MLX.ones(@normalized_shape)
          end
        end
      end
      
      # Batch Normalization
      class BatchNorm < MLX::NN::Module
        attr_reader :num_features, :eps, :momentum, :affine, :track_running_stats
        
        def initialize(num_features, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: true)
          super()
          
          @num_features = num_features
          @eps = eps
          @momentum = momentum
          @affine = affine
          @track_running_stats = track_running_stats
          
          if affine
            weight = MLX.ones([num_features])
            bias = MLX.zeros([num_features])
            
            register_parameter('weight', weight)
            register_parameter('bias', bias)
          end
          
          if track_running_stats
            register_parameter('running_mean', MLX.zeros([num_features]))
            register_parameter('running_var', MLX.ones([num_features]))
            register_parameter('num_batches_tracked', MLX.array(0, dtype: MLX::INT64))
          end
        end
        
        def forward(x)
          # Handle different input shapes (2D, 3D, 4D, 5D)
          dims = x.ndim
          if dims < 2
            raise ArgumentError, "BatchNorm expected at least 2D input, got #{dims}D"
          end
          
          if dims == 2
            # (batch_size, num_features)
            reduce_axes = [0]
            reshape_shape = [1, @num_features]
          elsif dims == 3
            # (batch_size, num_features, seq_len)
            reduce_axes = [0, 2]
            reshape_shape = [1, @num_features, 1]
          elsif dims == 4
            # (batch_size, num_features, height, width)
            reduce_axes = [0, 2, 3]
            reshape_shape = [1, @num_features, 1, 1]
          elsif dims == 5
            # (batch_size, num_features, depth, height, width)
            reduce_axes = [0, 2, 3, 4]
            reshape_shape = [1, @num_features, 1, 1, 1]
          end
          
          if @training || !@track_running_stats
            # Calculate batch statistics
            batch_mean = MLX.mean(x, axis: reduce_axes, keepdim: true)
            batch_var = MLX.mean(MLX.power(MLX.subtract(x, batch_mean), 2), axis: reduce_axes, keepdim: true)
            
            if @training && @track_running_stats
              # Update running statistics
              if @momentum.is_a?(Numeric)
                momentum_tensor = @momentum
              else
                # If momentum is None, use a cumulative moving average
                batches = @_parameters['num_batches_tracked'] + 1
                @_parameters['num_batches_tracked'] = batches
                momentum_tensor = 1.0 / batches
              end
              
              # Update running mean and variance
              running_mean = @_parameters['running_mean']
              running_var = @_parameters['running_var']
              
              flat_batch_mean = MLX.reshape(batch_mean, [@num_features])
              flat_batch_var = MLX.reshape(batch_var, [@num_features])
              
              new_running_mean = (1 - momentum_tensor) * running_mean + momentum_tensor * flat_batch_mean
              new_running_var = (1 - momentum_tensor) * running_var + momentum_tensor * flat_batch_var
              
              @_parameters['running_mean'] = new_running_mean
              @_parameters['running_var'] = new_running_var
            end
            
            # Use batch statistics for normalization
            mean_to_use = batch_mean
            var_to_use = batch_var
          else
            # Use running statistics for normalization
            running_mean = @_parameters['running_mean']
            running_var = @_parameters['running_var']
            
            # Reshape for broadcasting
            mean_to_use = MLX.reshape(running_mean, reshape_shape)
            var_to_use = MLX.reshape(running_var, reshape_shape)
          end
          
          # Normalize
          normalized = MLX.divide(MLX.subtract(x, mean_to_use), MLX.sqrt(MLX.add(var_to_use, @eps)))
          
          if @affine
            # Apply scale and bias
            weight = MLX.reshape(@_parameters['weight'], reshape_shape)
            bias = MLX.reshape(@_parameters['bias'], reshape_shape)
            
            normalized = MLX.add(MLX.multiply(normalized, weight), bias)
          end
          
          normalized
        end
        
        def reset_parameters
          if @affine
            @_parameters['weight'] = MLX.ones([@num_features])
            @_parameters['bias'] = MLX.zeros([@num_features])
          end
          
          if @track_running_stats
            @_parameters['running_mean'] = MLX.zeros([@num_features])
            @_parameters['running_var'] = MLX.ones([@num_features])
            @_parameters['num_batches_tracked'] = MLX.array(0, dtype: MLX::INT64)
          end
        end
      end
      
      # Group Normalization
      class GroupNorm < MLX::NN::Module
        attr_reader :num_groups, :num_channels, :eps, :affine
        
        def initialize(num_groups, num_channels, eps: 1e-5, affine: true)
          super()
          
          if num_channels % num_groups != 0
            raise ArgumentError, "num_channels (#{num_channels}) must be divisible by num_groups (#{num_groups})"
          end
          
          @num_groups = num_groups
          @num_channels = num_channels
          @eps = eps
          @affine = affine
          
          if affine
            weight = MLX.ones([num_channels])
            bias = MLX.zeros([num_channels])
            
            register_parameter('weight', weight)
            register_parameter('bias', bias)
          end
        end
        
        def forward(x)
          # Handle different input shapes
          dims = x.ndim
          if dims < 2
            raise ArgumentError, "GroupNorm expected at least 2D input, got #{dims}D"
          end
          
          # Get shape information
          batch_size = x.shape[0]
          
          # Reshape to separate groups: (batch_size, num_groups, channels_per_group, *rest)
          channels_per_group = @num_channels / @num_groups
          
          # Reshape for group normalization
          if dims == 2
            # (batch_size, num_channels) -> (batch_size, num_groups, channels_per_group)
            reshaped = MLX.reshape(x, [batch_size, @num_groups, channels_per_group])
            reduce_axes = [2]
            orig_shape = x.shape
          elsif dims == 3
            # (batch_size, num_channels, seq_len) -> (batch_size, num_groups, channels_per_group, seq_len)
            seq_len = x.shape[2]
            reshaped = MLX.reshape(x, [batch_size, @num_groups, channels_per_group, seq_len])
            reduce_axes = [2, 3]
            orig_shape = x.shape
          elsif dims == 4
            # (batch_size, num_channels, height, width) -> (batch_size, num_groups, channels_per_group, height, width)
            height, width = x.shape[2], x.shape[3]
            reshaped = MLX.reshape(x, [batch_size, @num_groups, channels_per_group, height, width])
            reduce_axes = [2, 3, 4]
            orig_shape = x.shape
          elsif dims == 5
            # (batch_size, num_channels, depth, height, width) -> (batch_size, num_groups, channels_per_group, depth, height, width)
            depth, height, width = x.shape[2], x.shape[3], x.shape[4]
            reshaped = MLX.reshape(x, [batch_size, @num_groups, channels_per_group, depth, height, width])
            reduce_axes = [2, 3, 4, 5]
            orig_shape = x.shape
          end
          
          # Calculate stats per group
          group_mean = MLX.mean(reshaped, axis: reduce_axes, keepdim: true)
          group_var = MLX.mean(MLX.power(MLX.subtract(reshaped, group_mean), 2), axis: reduce_axes, keepdim: true)
          
          # Normalize
          normalized = MLX.divide(MLX.subtract(reshaped, group_mean), MLX.sqrt(MLX.add(group_var, @eps)))
          
          # Reshape back to original dimensions
          normalized = MLX.reshape(normalized, orig_shape)
          
          if @affine
            # Per-channel scale and bias
            # Create broadcastable weight and bias tensors
            weight_shape = [1, @num_channels] + [1] * (dims - 2)
            bias_shape = [1, @num_channels] + [1] * (dims - 2)
            
            weight_reshaped = MLX.reshape(@_parameters['weight'], weight_shape)
            bias_reshaped = MLX.reshape(@_parameters['bias'], bias_shape)
            
            normalized = MLX.add(MLX.multiply(normalized, weight_reshaped), bias_reshaped)
          end
          
          normalized
        end
        
        def reset_parameters
          if @affine
            @_parameters['weight'] = MLX.ones([@num_channels])
            @_parameters['bias'] = MLX.zeros([@num_channels])
          end
        end
      end
      
      # Instance Normalization
      class InstanceNorm < MLX::NN::Module
        attr_reader :num_features, :eps, :affine, :track_running_stats, :momentum
        
        def initialize(num_features, eps: 1e-5, momentum: 0.1, affine: true, track_running_stats: false)
          super()
          
          @num_features = num_features
          @eps = eps
          @momentum = momentum
          @affine = affine
          @track_running_stats = track_running_stats
          
          if affine
            weight = MLX.ones([num_features])
            bias = MLX.zeros([num_features])
            
            register_parameter('weight', weight)
            register_parameter('bias', bias)
          end
          
          if track_running_stats
            register_parameter('running_mean', MLX.zeros([num_features]))
            register_parameter('running_var', MLX.ones([num_features]))
            register_parameter('num_batches_tracked', MLX.array(0, dtype: MLX::INT64))
          end
        end
        
        def forward(x)
          # Handle different input shapes
          dims = x.ndim
          if dims < 3
            raise ArgumentError, "InstanceNorm expected at least 3D input, got #{dims}D"
          end
          
          # Spatial dimensions to normalize over (all except batch and channel dims)
          reduce_axes = (2...dims).to_a
          
          if @training || !@track_running_stats
            # Calculate instance statistics (per-channel, per-sample stats)
            instance_mean = MLX.mean(x, axis: reduce_axes, keepdim: true)
            instance_var = MLX.mean(MLX.power(MLX.subtract(x, instance_mean), 2), axis: reduce_axes, keepdim: true)
            
            if @training && @track_running_stats
              # Update running statistics
              # First compute batch-averaged stats
              batch_mean = MLX.mean(instance_mean, axis: 0)
              batch_var = MLX.mean(instance_var, axis: 0)
              
              if @momentum.is_a?(Numeric)
                momentum_tensor = @momentum
              else
                # If momentum is None, use a cumulative moving average
                batches = @_parameters['num_batches_tracked'] + 1
                @_parameters['num_batches_tracked'] = batches
                momentum_tensor = 1.0 / batches
              end
              
              # Update running mean and variance
              running_mean = @_parameters['running_mean']
              running_var = @_parameters['running_var']
              
              new_running_mean = (1 - momentum_tensor) * running_mean + momentum_tensor * batch_mean
              new_running_var = (1 - momentum_tensor) * running_var + momentum_tensor * batch_var
              
              @_parameters['running_mean'] = new_running_mean
              @_parameters['running_var'] = new_running_var
            end
            
            # Use instance statistics for normalization
            mean_to_use = instance_mean
            var_to_use = instance_var
          else
            # Use running statistics for normalization
            running_mean = @_parameters['running_mean']
            running_var = @_parameters['running_var']
            
            # Reshape for broadcasting
            reshape_shape = [1, @num_features] + [1] * (dims - 2)
            mean_to_use = MLX.reshape(running_mean, reshape_shape)
            var_to_use = MLX.reshape(running_var, reshape_shape)
          end
          
          # Normalize
          normalized = MLX.divide(MLX.subtract(x, mean_to_use), MLX.sqrt(MLX.add(var_to_use, @eps)))
          
          if @affine
            # Apply scale and bias
            weight_shape = [1, @num_features] + [1] * (dims - 2)
            bias_shape = [1, @num_features] + [1] * (dims - 2)
            
            weight_reshaped = MLX.reshape(@_parameters['weight'], weight_shape)
            bias_reshaped = MLX.reshape(@_parameters['bias'], bias_shape)
            
            normalized = MLX.add(MLX.multiply(normalized, weight_reshaped), bias_reshaped)
          end
          
          normalized
        end
        
        def reset_parameters
          if @affine
            @_parameters['weight'] = MLX.ones([@num_features])
            @_parameters['bias'] = MLX.zeros([@num_features])
          end
          
          if @track_running_stats
            @_parameters['running_mean'] = MLX.zeros([@num_features])
            @_parameters['running_var'] = MLX.ones([@num_features])
            @_parameters['num_batches_tracked'] = MLX.array(0, dtype: MLX::INT64)
          end
        end
      end
    end
  end
end 