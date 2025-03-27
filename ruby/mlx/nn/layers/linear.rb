module MLX
  module NN
    module Layers
      # Linear layer (fully connected layer)
      class Linear < MLX::NN::Module
        attr_reader :in_features, :out_features, :bias
        
        # Initialize a linear layer
        # @param in_features [Integer] size of each input sample
        # @param out_features [Integer] size of each output sample
        # @param bias [Boolean] whether to include a bias term
        def initialize(in_features, out_features, bias: true)
          super()
          @in_features = in_features
          @out_features = out_features
          @bias = bias
          
          # Initialize weights
          weight = MLX::NN::Init.kaiming_uniform([in_features, out_features], Ops.sqrt(5))
          register_parameter('weight', weight)
          
          if bias
            # Initialize bias
            bound = 1.0 / Ops.sqrt(in_features)
            bias_tensor = MLX::NN::Init.uniform([out_features], -bound, bound)
            register_parameter('bias', bias_tensor)
          end
        end
        
        # Forward pass
        # @param x [MLX::Array] input tensor of shape (*, in_features)
        # @return [MLX::Array] output tensor of shape (*, out_features)
        def forward(x)
          output = MLX.matmul(x, @_parameters['weight'])
          
          if @bias
            output = MLX.add(output, @_parameters['bias'])
          end
          
          output
        end
        
        # Reset parameters to their initial values
        def reset_parameters
          fan_in, _ = MLX::NN::Init.compute_fans(@_parameters['weight'].shape)
          bound = 1.0 / Ops.sqrt(fan_in)
          
          # Reset weight
          weight = MLX::NN::Init.kaiming_uniform([@in_features, @out_features], Ops.sqrt(5))
          @_parameters['weight'] = weight
          
          # Reset bias if present
          if @bias
            bias_tensor = MLX::NN::Init.uniform([@out_features], -bound, bound)
            @_parameters['bias'] = bias_tensor
          end
        end
      end
      
      # Identity layer (does nothing to the input)
      class Identity < MLX::NN::Module
        def forward(x)
          x
        end
      end
      
      # Bilinear layer
      class Bilinear < MLX::NN::Module
        attr_reader :in1_features, :in2_features, :out_features, :bias
        
        # Initialize a bilinear layer
        # @param in1_features [Integer] size of first input sample
        # @param in2_features [Integer] size of second input sample
        # @param out_features [Integer] size of each output sample
        # @param bias [Boolean] whether to include a bias term
        def initialize(in1_features, in2_features, out_features, bias: true)
          super()
          @in1_features = in1_features
          @in2_features = in2_features
          @out_features = out_features
          @bias = bias
          
          # Initialize weights
          bound = 1.0 / Ops.sqrt(in1_features * in2_features)
          weight = MLX::NN::Init.uniform([out_features, in1_features, in2_features], -bound, bound)
          register_parameter('weight', weight)
          
          if bias
            # Initialize bias
            bias_tensor = MLX::NN::Init.uniform([out_features], -bound, bound)
            register_parameter('bias', bias_tensor)
          end
        end
        
        # Forward pass
        # @param x1 [MLX::Array] first input tensor of shape (batch_size, in1_features)
        # @param x2 [MLX::Array] second input tensor of shape (batch_size, in2_features)
        # @return [MLX::Array] output tensor of shape (batch_size, out_features)
        def forward(x1, x2)
          # Check input dimensions
          batch_size = x1.shape[0]
          
          # Reshape x1 for batched matrix multiplication: (batch_size, 1, in1_features)
          x1_reshaped = MLX.reshape(x1, [batch_size, 1, @in1_features])
          
          # Reshape x2 for batched matrix multiplication: (batch_size, in2_features, 1)
          x2_reshaped = MLX.reshape(x2, [batch_size, @in2_features, 1])
          
          # Perform batched matrix multiplication: (batch_size, 1, in2_features)
          batched_x1x2 = MLX.matmul(x1_reshaped, x2_reshaped)
          
          # Reshape result: (batch_size, in1_features * in2_features)
          batched_x1x2 = MLX.reshape(batched_x1x2, [batch_size, @in1_features * @in2_features])
          
          # Reshape weight for matrix multiplication: (out_features, in1_features * in2_features)
          weight_reshaped = MLX.reshape(@_parameters['weight'], [@out_features, @in1_features * @in2_features])
          
          # Perform matrix multiplication: (batch_size, out_features)
          output = MLX.matmul(batched_x1x2, MLX.transpose(weight_reshaped))
          
          if @bias
            output = MLX.add(output, @_parameters['bias'])
          end
          
          output
        end
        
        # Reset parameters to their initial values
        def reset_parameters
          bound = 1.0 / Ops.sqrt(@in1_features * @in2_features)
          
          # Reset weight
          weight = MLX::NN::Init.uniform([@out_features, @in1_features, @in2_features], -bound, bound)
          @_parameters['weight'] = weight
          
          # Reset bias if present
          if @bias
            bias_tensor = MLX::NN::Init.uniform([@out_features], -bound, bound)
            @_parameters['bias'] = bias_tensor
          end
        end
      end
    end
  end
end 