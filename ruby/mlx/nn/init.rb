module MLX
  module NN
    module Init
      # Xavier/Glorot uniform initialization
      def self.xavier_uniform(shape, gain = 1.0)
        fan_in, fan_out = compute_fans(shape)
        bound = gain * Math.sqrt(6.0 / (fan_in + fan_out))
        MLX.random_uniform(-bound, bound, shape)
      end
      
      # Xavier/Glorot normal initialization
      def self.xavier_normal(shape, gain = 1.0)
        fan_in, fan_out = compute_fans(shape)
        std = gain * Math.sqrt(2.0 / (fan_in + fan_out))
        MLX.random_normal(0.0, std, shape)
      end
      
      # Kaiming/He uniform initialization
      def self.kaiming_uniform(shape, gain = 1.0, mode = 'fan_in', nonlinearity = 'relu')
        fan_in, fan_out = compute_fans(shape)
        fan = (mode == 'fan_in') ? fan_in : fan_out
        gain = calculate_gain(nonlinearity, gain)
        bound = gain * Math.sqrt(3.0 / fan)
        MLX.random_uniform(-bound, bound, shape)
      end
      
      # Kaiming/He normal initialization
      def self.kaiming_normal(shape, gain = 1.0, mode = 'fan_in', nonlinearity = 'relu')
        fan_in, fan_out = compute_fans(shape)
        fan = (mode == 'fan_in') ? fan_in : fan_out
        gain = calculate_gain(nonlinearity, gain)
        std = gain / Math.sqrt(fan)
        MLX.random_normal(0.0, std, shape)
      end
      
      # Uniform initialization
      def self.uniform(shape, lower = -0.1, upper = 0.1)
        MLX.random_uniform(lower, upper, shape)
      end
      
      # Normal initialization
      def self.normal(shape, mean = 0.0, std = 0.1)
        MLX.random_normal(mean, std, shape)
      end
      
      # Constant initialization
      def self.constant(shape, value = 0.0)
        MLX.full(shape, value)
      end
      
      # Identity initialization (for square matrices)
      def self.identity(shape)
        if shape.length != 2 || shape[0] != shape[1]
          raise ArgumentError, "Identity initialization requires a square matrix"
        end
        MLX.identity(shape[0])
      end
      
      # Orthogonal initialization
      def self.orthogonal(shape, gain = 1.0)
        if shape.length < 2
          raise ArgumentError, "Orthogonal initialization requires at least 2 dimensions"
        end
        
        # Flatten to a 2D matrix
        rows = shape[0]
        cols = shape[1..-1].reduce(:*)
        flattened_shape = [rows, cols]
        
        # Generate random matrix
        key = MLX::Random.key(0)
        random_mat = MLX::Random.normal(key, flattened_shape)
        
        # Apply QR decomposition
        q, r = MLX::Linalg.qr(random_mat)
        
        # Make deterministic by adjusting signs
        d = MLX.diag_part(r)
        ph = MLX.sign(d)
        q = MLX.multiply(q, ph.reshape([1, -1]))
        
        # Reshape back to original shape and apply gain
        q = MLX.multiply(q, gain)
        
        if shape.length > 2
          q = MLX.reshape(q, shape)
        end
        
        q
      end

      # Sparse initialization
      def self.sparse(shape, sparsity = 0.9, std = 0.01)
        key = MLX::Random.key(0)
        w = MLX::Random.normal(key, shape, MLX::FLOAT32)
        
        # Create mask for sparsity
        mask_key = MLX::Random.split(key, 1)[0]
        mask = MLX::Random.uniform(mask_key, shape) > sparsity
        mask = MLX.convert(mask, MLX::FLOAT32)
        
        # Apply mask to weights
        w = MLX.multiply(w, mask) * std
        
        w
      end
      
      # Zero initialization
      def self.zeros(shape)
        MLX.zeros(shape)
      end
      
      # One initialization
      def self.ones(shape)
        MLX.ones(shape)
      end
      
      # Helper functions
      
      # Calculate fan_in and fan_out for a tensor shape
      def self.compute_fans(shape)
        if shape.length < 2
          fan_in = fan_out = shape.reduce(1, :*)
        elsif shape.length == 2
          fan_in, fan_out = shape
        else
          # For conv layers
          receptive_field_size = shape[2..-1].reduce(1, :*)
          fan_in = shape[1] * receptive_field_size
          fan_out = shape[0] * receptive_field_size
        end
        
        [fan_in, fan_out]
      end
      
      # Calculate gain for different activation functions
      def self.calculate_gain(nonlinearity, param = nil)
        case nonlinearity.to_s.downcase
        when 'linear', 'identity'
          1.0
        when 'sigmoid'
          1.0
        when 'tanh'
          5.0 / 3.0
        when 'relu'
          Math.sqrt(2.0)
        when 'leaky_relu'
          negative_slope = param || 0.01
          Math.sqrt(2.0 / (1 + negative_slope ** 2))
        when 'selu'
          1.0  # SELU should use 1.0 as recommended
        else
          1.0
        end
      end
    end
  end
end 