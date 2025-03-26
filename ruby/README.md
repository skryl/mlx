# MLX Ruby

Ruby bindings for Apple's MLX framework, providing a fast and efficient machine learning library that runs natively on Apple Silicon.

## Overview

MLX Ruby is a clean and idiomatic Ruby interface to the MLX framework, allowing Ruby developers to take advantage of Apple's high-performance machine learning library. With MLX Ruby, you can:

- Build and train neural networks directly in Ruby
- Leverage Apple Silicon's hardware acceleration
- Use a familiar API that closely mirrors PyTorch's design
- Execute operations lazily and compile them JIT
- Take advantage of MLX's automatic differentiation

## Installation

### Prerequisites

- macOS with Apple Silicon hardware (M1/M2/M3 chip)
- Ruby 3.0 or newer
- C++ compiler with C++17 support

### Installing from RubyGems

```bash
gem install mlx-ruby
```

### Building from source

```bash
git clone https://github.com/skryl/mlx-ruby.git
cd mlx-ruby
bundle install
rake compile
rake install
```

## Quick Start

```ruby
require 'mlx'

# Create tensors
x = MLX.array([1, 2, 3, 4], dtype: MLX::FLOAT32)
y = MLX.ones([4], dtype: MLX::FLOAT32)

# Basic operations
z = x + y                   # => [2, 3, 4, 5]
z = MLX.multiply(x, 2)      # => [2, 4, 6, 8]

# Create a simple neural network
class SimpleNN < MLX::NN::Module
  def initialize
    super()
    @linear1 = MLX::NN::Layers::Linear.new(4, 8)
    @activation = MLX::NN::Layers::ReLU.new
    @linear2 = MLX::NN::Layers::Linear.new(8, 1)
    
    register_module("linear1", @linear1)
    register_module("activation", @activation)
    register_module("linear2", @linear2)
  end
  
  def forward(x)
    x = @linear1.forward(x)
    x = @activation.forward(x)
    x = @linear2.forward(x)
    x
  end
end

# Instantiate the model
model = SimpleNN.new

# Get trainable parameters
params = model.trainable_parameters

# Create input data
batch_size = 32
x = MLX.random_normal(0, 1, [batch_size, 4])
y = MLX.random_normal(0, 1, [batch_size, 1])

# Define loss function
loss_fn = MLX::NN::Loss::MSELoss.new

# Create optimizer
optimizer = MLX::NN::Optim::Adam.new(params, lr: 0.01)

# Training loop
10.times do |epoch|
  # Forward pass
  y_pred = model.forward(x)
  loss = loss_fn.forward(y_pred, y)
  
  # Print loss
  puts "Epoch #{epoch}, Loss: #{MLX.to_ruby(loss)}"
  
  # Zero gradients
  optimizer.zero_grad
  
  # Backward pass (compute gradients)
  MLX.grad(loss, params)
  
  # Update parameters
  optimizer.step
end

# Save trained model
MLX.save(model, "simple_model.safetensors")
```

## Key Components

MLX Ruby provides a comprehensive set of components for building and training neural networks:

### Core Components

- `MLX::Array`: Multidimensional arrays (tensors)
- `MLX::Device`: Device management (CPU/GPU)
- `MLX::Stream`: Execution stream management
- `MLX::Random`: Random number generation
- `MLX::Linalg`: Linear algebra operations

### Neural Network Building Blocks

- `MLX::NN::Module`: Base class for all neural network modules
- `MLX::NN::Layers`: Various layer implementations
  - Linear layers (Linear, Bilinear)
  - Convolutional layers (Conv1d, Conv2d, Conv3d)
  - Pooling layers (MaxPool, AvgPool)
  - Normalization layers (BatchNorm, LayerNorm, etc.)
  - Recurrent layers (RNN, LSTM, GRU)
  - Transformer layers (MultiheadAttention, TransformerEncoder)
  - Activation functions (ReLU, Sigmoid, GELU, etc.)
- `MLX::NN::Loss`: Loss functions (MSE, CrossEntropy, etc.)
- `MLX::NN::Optim`: Optimizers (SGD, Adam, AdamW, etc.)
- `MLX::NN::LRScheduler`: Learning rate schedulers

## Advanced Usage

### Automatic Differentiation

MLX Ruby provides built-in automatic differentiation:

```ruby
x = MLX.array([1.0, 2.0, 3.0])
y = MLX.multiply(x, 2)

# Compute gradients
grads = MLX.grad(y.sum, x)
```

### Custom Layers

You can create custom neural network layers by subclassing `MLX::NN::Module`:

```ruby
class MyCustomLayer < MLX::NN::Module
  def initialize(in_features, out_features)
    super()
    @weight = MLX::NN::Init.kaiming_uniform([in_features, out_features])
    register_parameter('weight', @weight)
  end
  
  def forward(x)
    # Custom forward computation
    MLX.matmul(x, @weight)
  end
end
```

### GPU Acceleration

MLX automatically uses Metal for acceleration on Apple Silicon:

```ruby
# Get current device
device = MLX.device
puts "Current device: #{device}"

# Set default device
MLX.set_default_device('gpu')
```

## Documentation

For complete API documentation, please visit:

[https://mlx-ruby.github.io/mlx-ruby/](https://mlx-ruby.github.io/mlx-ruby/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Apple MLX team for creating the original framework
- The Ruby community for their continued support 