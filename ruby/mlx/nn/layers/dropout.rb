module MLX
  module NN
    module Layers
      # Base Dropout layer
      class Dropout < MLX::NN::Module
        attr_reader :p, :training
        
        def initialize(p = 0.5)
          super()
          @p = p
          @training = true
        end
        
        def forward(x)
          return x unless @training
          return x if @p == 0
          
          # Generate dropout mask
          mask = MLX::Random.bernoulli(1.0 - @p, x.shape)
          # Scale to maintain expected value
          scale = 1.0 / (1.0 - @p)
          x * mask * scale
        end
      end
      
      # 2D Dropout layer (for 4D inputs: batch_size, channels, height, width)
      class Dropout2d < MLX::NN::Module
        attr_reader :p, :training
        
        def initialize(p = 0.5)
          super()
          @p = p
          @training = true
        end
        
        def forward(x)
          return x unless @training
          return x if @p == 0
          
          # Dropout entire channels
          # For input shape [batch_size, channels, height, width]
          # Create mask of shape [batch_size, channels, 1, 1]
          shape = x.shape
          mask_shape = [shape[0], shape[1], 1, 1]
          mask = MLX::Random.bernoulli(1.0 - @p, mask_shape)
          # Scale to maintain expected value
          scale = 1.0 / (1.0 - @p)
          x * mask * scale
        end
      end
      
      # 3D Dropout layer (for 5D inputs: batch_size, channels, depth, height, width)
      class Dropout3d < MLX::NN::Module
        attr_reader :p, :training
        
        def initialize(p = 0.5)
          super()
          @p = p
          @training = true
        end
        
        def forward(x)
          return x unless @training
          return x if @p == 0
          
          # Dropout entire channels
          # For input shape [batch_size, channels, depth, height, width]
          # Create mask of shape [batch_size, channels, 1, 1, 1]
          shape = x.shape
          mask_shape = [shape[0], shape[1], 1, 1, 1]
          mask = MLX::Random.bernoulli(1.0 - @p, mask_shape)
          # Scale to maintain expected value
          scale = 1.0 / (1.0 - @p)
          x * mask * scale
        end
      end
    end
  end
end 