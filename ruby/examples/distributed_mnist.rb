#!/usr/bin/env ruby
# Example of distributed training for MNIST dataset using MLX Ruby bindings

require 'mlx'

class MNISTModel < MLX::NN::Module
  def initialize
    super()
    
    # Define a simple convolutional network
    @layers = MLX::NN::Sequential.new(
      MLX::NN::Layers::Conv2d.new(1, 32, kernel_size: 3, stride: 1, padding: 1),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::MaxPool2d.new(kernel_size: 2, stride: 2),
      MLX::NN::Layers::Conv2d.new(32, 64, kernel_size: 3, stride: 1, padding: 1),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::MaxPool2d.new(kernel_size: 2, stride: 2),
      MLX::NN::Layers::Flatten.new,
      MLX::NN::Layers::Linear.new(7*7*64, 128),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::Linear.new(128, 10)
    )
  end
  
  def forward(x)
    @layers.call(x)
  end
end

# Initialize distributed environment
world = MLX::Distributed.init()
rank = world.rank
size = world.size

puts "Running on rank #{rank} of #{size}"

# Create model
model = MNISTModel.new

# Convert the last layer to a distributed layer if running in distributed mode
if size > 1
  # Replace the last linear layer with a distributed version
  last_linear = model.layers[-1]
  distributed_linear = MLX::NN::Layers.sharded_to_all_linear(last_linear, group: world)
  model.layers[-1] = distributed_linear
  
  puts "Using distributed linear layer for output on rank #{rank}"
end

# Loss function
loss_fn = lambda do |model, x, y|
  logits = model.call(x)
  loss = MLX::NN.losses.cross_entropy(logits, y)
  
  # Average loss across all processes
  if size > 1
    loss = MLX::Distributed.all_sum(loss) / size
  end
  
  loss
end

# Create optimizer
optimizer = MLX::Optimizers::Adam.new(learning_rate: 0.001)

# Training step
train_step = lambda do |model, x, y|
  # Forward and backward to compute gradients
  loss, grads = MLX.value_and_grad(model, loss_fn).call(model, x, y)
  
  # Update model using gradients
  optimizer.update(model, grads)
  
  loss
end

# Evaluation function
eval_fn = lambda do |model, x, y|
  logits = model.call(x)
  preds = MLX.argmax(logits, axis: 1)
  accuracy = MLX.mean(preds.equal(y))
  
  # Average accuracy across all processes
  if size > 1
    accuracy = MLX::Distributed.all_sum(accuracy) / size
  end
  
  accuracy
end

# Mock training data (in a real scenario, we would load MNIST dataset)
# Each process gets a subset of the data
batch_size = 64
num_batches = 10

if rank == 0
  puts "Starting training with #{size} processes"
end

# Training loop
10.times do |epoch|
  epoch_loss = 0.0
  
  num_batches.times do |batch|
    # Create mock data (simulating MNIST)
    x = MLX.random.normal([batch_size, 1, 28, 28])
    y = MLX.random.randint(0, 10, [batch_size]).astype(MLX::INT32)
    
    # Train step
    loss = train_step.call(model, x, y)
    epoch_loss += loss.item
  end
  
  epoch_loss /= num_batches
  
  # Evaluate (using training data as mock test data)
  x = MLX.random.normal([batch_size, 1, 28, 28])
  y = MLX.random.randint(0, 10, [batch_size]).astype(MLX::INT32)
  accuracy = eval_fn.call(model, x, y)
  
  # Only print from rank 0 to avoid duplicate output
  if rank == 0
    puts "Epoch #{epoch}, Loss: #{epoch_loss.round(4)}, Accuracy: #{(accuracy.item * 100).round(2)}%"
  end
end

puts "Training complete on rank #{rank}" 