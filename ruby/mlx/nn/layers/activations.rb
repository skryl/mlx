module MLX
  module NN
    module Layers
      # Base activation function methods
      module ActivationFunctions
        # ReLU activation: max(0, x)
        def self.relu(x)
          MLX.maximum(x, 0)
        end
        
        # Leaky ReLU activation: max(0, x) + negative_slope * min(0, x)
        def self.leaky_relu(x, negative_slope = 0.01)
          MLX.maximum(x, 0) + negative_slope * MLX.minimum(x, 0)
        end
        
        # Sigmoid activation: 1 / (1 + exp(-x))
        def self.sigmoid(x)
          MLX.sigmoid(x)
        end
        
        # Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        def self.tanh(x)
          MLX.tanh(x)
        end
        
        # Softmax activation: exp(x) / sum(exp(x), axis)
        def self.softmax(x, axis = -1)
          MLX.softmax(x, axis)
        end
        
        # Log softmax activation: log(softmax(x))
        def self.log_softmax(x, axis = -1)
          # Numerically stable version
          x_max = MLX.max(x, axis: axis, keepdim: true)
          x_diff = MLX.subtract(x, x_max)
          exp_x = MLX.exp(x_diff)
          sum_exp = MLX.sum(exp_x, axis: axis, keepdim: true)
          log_sum = MLX.log(sum_exp)
          result = MLX.subtract(x_diff, log_sum)
          result
        end
        
        # GELU activation: x * Φ(x) where Φ is the CDF of the standard normal distribution
        def self.gelu(x)
          # Approximation using tanh
          sqrt_2_over_pi = Math.sqrt(2.0 / Math::PI)
          inner = sqrt_2_over_pi * (x + 0.044715 * MLX.power(x, 3))
          x * 0.5 * (1 + MLX.tanh(inner))
        end
        
        # Fast GELU approximation
        def self.gelu_fast_approx(x)
          x * MLX.sigmoid(1.702 * x)
        end
        
        # SiLU (Swish) activation: x * sigmoid(x)
        def self.silu(x)
          x * MLX.sigmoid(x)
        end
        
        # ELU activation: x if x > 0 else alpha * (exp(x) - 1)
        def self.elu(x, alpha = 1.0)
          MLX.maximum(x, 0) + alpha * (MLX.exp(MLX.minimum(x, 0)) - 1)
        end
        
        # SELU activation (Self-Normalized)
        def self.selu(x)
          alpha = 1.6732632423543772848170429916717
          scale = 1.0507009873554804934193349852946
          scale * (MLX.maximum(x, 0) + alpha * (MLX.exp(MLX.minimum(x, 0)) - 1))
        end
        
        # CELU activation
        def self.celu(x, alpha = 1.0)
          MLX.maximum(x, 0) + alpha * (MLX.exp(MLX.minimum(x, 0) / alpha) - 1)
        end
        
        # Mish activation: x * tanh(softplus(x))
        def self.mish(x)
          x * MLX.tanh(MLX.log(1 + MLX.exp(x)))
        end
        
        # Softplus activation: log(1 + exp(x))
        def self.softplus(x, beta = 1.0, threshold = 20.0)
          # Numerically stable version
          result = MLX.zeros_like(x)
          mask = (x * beta) <= threshold
          safe_x = MLX.log(1 + MLX.exp(x * beta)) / beta
          result = MLX.where(mask, safe_x, x)
          result
        end
        
        # ReLU6 activation: min(max(0, x), 6)
        def self.relu6(x)
          MLX.minimum(MLX.maximum(x, 0), 6)
        end
        
        # Hardtanh activation: -1 if x < -1, 1 if x > 1, x otherwise
        def self.hard_tanh(x, min_val = -1.0, max_val = 1.0)
          MLX.minimum(MLX.maximum(x, min_val), max_val)
        end
        
        # Hardswish activation: x * (ReLU6(x + 3) / 6)
        def self.hardswish(x)
          x * (MLX.minimum(MLX.maximum(x + 3, 0), 6) / 6)
        end
        
        # GLU (Gated Linear Unit): (x_a * sigmoid(x_b)) where x is split along dim
        def self.glu(x, dim = -1)
          # Split x into two parts
          x_dim = x.shape[dim]
          x_a, x_b = MLX.split(x, 2, dim)
          x_a * MLX.sigmoid(x_b)
        end
      end
      
      # ReLU activation module
      class ReLU < MLX::NN::Module
        def forward(x)
          ActivationFunctions.relu(x)
        end
      end
      
      # Leaky ReLU activation module
      class LeakyReLU < MLX::NN::Module
        attr_reader :negative_slope
        
        def initialize(negative_slope = 0.01)
          super()
          @negative_slope = negative_slope
        end
        
        def forward(x)
          ActivationFunctions.leaky_relu(x, @negative_slope)
        end
      end
      
      # Sigmoid activation module
      class Sigmoid < MLX::NN::Module
        def forward(x)
          ActivationFunctions.sigmoid(x)
        end
      end
      
      # Tanh activation module
      class Tanh < MLX::NN::Module
        def forward(x)
          ActivationFunctions.tanh(x)
        end
      end
      
      # Softmax activation module
      class Softmax < MLX::NN::Module
        attr_reader :dim
        
        def initialize(dim = -1)
          super()
          @dim = dim
        end
        
        def forward(x)
          ActivationFunctions.softmax(x, @dim)
        end
      end
      
      # LogSoftmax activation module
      class LogSoftmax < MLX::NN::Module
        attr_reader :dim
        
        def initialize(dim = -1)
          super()
          @dim = dim
        end
        
        def forward(x)
          ActivationFunctions.log_softmax(x, @dim)
        end
      end
      
      # GELU activation module
      class GELU < MLX::NN::Module
        def forward(x)
          ActivationFunctions.gelu(x)
        end
      end
      
      # SiLU activation module
      class SiLU < MLX::NN::Module
        def forward(x)
          ActivationFunctions.silu(x)
        end
      end
      
      # ELU activation module
      class ELU < MLX::NN::Module
        attr_reader :alpha
        
        def initialize(alpha = 1.0)
          super()
          @alpha = alpha
        end
        
        def forward(x)
          ActivationFunctions.elu(x, @alpha)
        end
      end
      
      # SELU activation module
      class SELU < MLX::NN::Module
        def forward(x)
          ActivationFunctions.selu(x)
        end
      end
      
      # CELU activation module
      class CELU < MLX::NN::Module
        attr_reader :alpha
        
        def initialize(alpha = 1.0)
          super()
          @alpha = alpha
        end
        
        def forward(x)
          ActivationFunctions.celu(x, @alpha)
        end
      end
      
      # Mish activation module
      class Mish < MLX::NN::Module
        def forward(x)
          ActivationFunctions.mish(x)
        end
      end
      
      # ReLU6 activation module
      class ReLU6 < MLX::NN::Module
        def forward(x)
          ActivationFunctions.relu6(x)
        end
      end
      
      # Hardtanh activation module
      class HardTanh < MLX::NN::Module
        attr_reader :min_val, :max_val
        
        def initialize(min_val = -1.0, max_val = 1.0)
          super()
          @min_val = min_val
          @max_val = max_val
        end
        
        def forward(x)
          ActivationFunctions.hard_tanh(x, @min_val, @max_val)
        end
      end
      
      # Hardswish activation module
      class Hardswish < MLX::NN::Module
        def forward(x)
          ActivationFunctions.hardswish(x)
        end
      end
      
      # GLU activation module
      class GLU < MLX::NN::Module
        attr_reader :dim
        
        def initialize(dim = -1)
          super()
          @dim = dim
        end
        
        def forward(x)
          ActivationFunctions.glu(x, @dim)
        end
      end
      
      # PReLU activation module (learnable parameters)
      class PReLU < MLX::NN::Module
        attr_reader :num_parameters, :init
        
        def initialize(num_parameters = 1, init = 0.25)
          super()
          @num_parameters = num_parameters
          @init = init
          
          # Initialize weight (learnable negative slope)
          weight = MLX.full([num_parameters], init)
          register_parameter('weight', weight)
        end
        
        def forward(x)
          # Apply PReLU: max(0, x) + weight * min(0, x)
          weight = @_parameters['weight']
          
          if @num_parameters == 1
            # Scalar case - same negative slope for all channels
            MLX.maximum(x, 0) + weight * MLX.minimum(x, 0)
          else
            # Per-channel case - reshape weight for broadcasting
            shape = [1] * x.ndim
            shape[1] = @num_parameters  # Assuming NCHW format
            
            weight_broadcasted = MLX.reshape(weight, shape)
            MLX.maximum(x, 0) + weight_broadcasted * MLX.minimum(x, 0)
          end
        end
        
        def reset_parameters
          @_parameters['weight'] = MLX.full([@num_parameters], @init)
        end
      end
      
      # Softmin activation module
      class Softmin < MLX::NN::Module
        attr_reader :dim
        
        def initialize(dim = -1)
          super()
          @dim = dim
        end
        
        def forward(x)
          ActivationFunctions.softmax(-x, @dim)
        end
      end
      
      # Log Sigmoid activation module
      class LogSigmoid < MLX::NN::Module
        def forward(x)
          -MLX.softplus(-x)
        end
      end
      
      # Softplus activation module
      class Softplus < MLX::NN::Module
        attr_reader :beta, :threshold
        
        def initialize(beta = 1.0, threshold = 20.0)
          super()
          @beta = beta
          @threshold = threshold
        end
        
        def forward(x)
          ActivationFunctions.softplus(x, @beta, @threshold)
        end
      end
      
      # Softsign activation module: x / (1 + |x|)
      class Softsign < MLX::NN::Module
        def forward(x)
          x / (1 + MLX.abs(x))
        end
      end
      
      # Softshrink activation module
      class Softshrink < MLX::NN::Module
        attr_reader :lambda
        
        def initialize(lambda_val = 0.5)
          super()
          @lambda = lambda_val
        end
        
        def forward(x)
          # x - lambda if x > lambda
          # x + lambda if x < -lambda
          # 0 otherwise
          positive_part = MLX.maximum(0, x - @lambda)
          negative_part = MLX.minimum(0, x + @lambda)
          positive_part + negative_part
        end
      end
      
      # Hardshrink activation module
      class HardShrink < MLX::NN::Module
        attr_reader :lambda
        
        def initialize(lambda_val = 0.5)
          super()
          @lambda = lambda_val
        end
        
        def forward(x)
          # x if x > lambda or x < -lambda
          # 0 otherwise
          mask = MLX.logical_or(x > @lambda, x < -@lambda)
          MLX.where(mask, x, 0)
        end
      end
      
      # Step activation module
      class Step < MLX::NN::Module
        def forward(x)
          MLX.greater_equal(x, 0).astype(x.dtype)
        end
      end
    end
  end
end 