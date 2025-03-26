require_relative 'mlx_test_case'

class TestOptimizers < MLXTestCase
  def setup
    super
    
    # Create a simple model for optimizer tests
    class SimpleModel < MLX::NN::Module
      attr_reader :linear1, :linear2
      
      def initialize
        super()
        @linear1 = MLX::NN::Layers::Linear.new(10, 20)
        @linear2 = MLX::NN::Layers::Linear.new(20, 1)
      end
      
      def forward(x)
        x = MLX.relu(@linear1.call(x))
        @linear2.call(x)
      end
    end
    
    # Create model, input, and target
    @model = SimpleModel.new
    @x = MLX.random.normal([16, 10])
    @y = MLX.random.normal([16, 1])
    
    # Define loss function
    @loss_fn = lambda do |model, x, y|
      preds = model.call(x)
      MLX.mean(MLX.square(preds - y))
    end
  end
  
  def test_sgd_optimizer
    # Create SGD optimizer
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.01)
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
  end
  
  def test_sgd_momentum
    # Create SGD with momentum
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.01, momentum: 0.9)
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
    assert optimizer.state.key?("momentum_buffers")
    
    # Do another update to check momentum
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    optimizer.update(@model, grads)
  end
  
  def test_adam_optimizer
    # Create Adam optimizer
    optimizer = MLX::Optimizers::Adam.new(
      learning_rate: 0.001, 
      betas: [0.9, 0.999], 
      eps: 1e-8
    )
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
    assert optimizer.state.key?("exp_avg")
    assert optimizer.state.key?("exp_avg_sq")
  end
  
  def test_adamw_optimizer
    # Create AdamW optimizer
    optimizer = MLX::Optimizers::AdamW.new(
      learning_rate: 0.001, 
      betas: [0.9, 0.999], 
      eps: 1e-8,
      weight_decay: 0.01
    )
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
    assert optimizer.state.key?("exp_avg")
    assert optimizer.state.key?("exp_avg_sq")
  end
  
  def test_rmsprop_optimizer
    # Create RMSprop optimizer
    optimizer = MLX::Optimizers::RMSprop.new(
      learning_rate: 0.01, 
      alpha: 0.99,
      eps: 1e-8
    )
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
    assert optimizer.state.key?("square_avg")
  end
  
  def test_adagrad_optimizer
    # Create Adagrad optimizer
    optimizer = MLX::Optimizers::Adagrad.new(
      learning_rate: 0.01, 
      eps: 1e-8
    )
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
    assert optimizer.state.key?("sum")
  end
  
  def test_adamax_optimizer
    # Skip if Adamax is not implemented
    skip unless defined?(MLX::Optimizers::Adamax)
    
    # Create Adamax optimizer
    optimizer = MLX::Optimizers::Adamax.new(
      learning_rate: 0.001, 
      betas: [0.9, 0.999], 
      eps: 1e-8
    )
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
    assert optimizer.state.key?("exp_avg")
    assert optimizer.state.key?("exp_inf")
  end
  
  def test_adadelta_optimizer
    # Skip if Adadelta is not implemented
    skip unless defined?(MLX::Optimizers::Adadelta)
    
    # Create Adadelta optimizer
    optimizer = MLX::Optimizers::Adadelta.new(
      learning_rate: 1.0, 
      rho: 0.9, 
      eps: 1e-6
    )
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
    assert optimizer.state.key?("square_avg")
    assert optimizer.state.key?("acc_delta")
  end
  
  def test_lion_optimizer
    # Skip if Lion is not implemented
    skip unless defined?(MLX::Optimizers::Lion)
    
    # Create Lion optimizer
    optimizer = MLX::Optimizers::Lion.new(
      learning_rate: 0.0001, 
      betas: [0.9, 0.99],
      weight_decay: 0.0
    )
    
    # Compute loss and gradients
    loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
    
    # Store original parameters
    old_params = @model.parameters.transform_values(&:copy)
    
    # Update parameters
    optimizer.update(@model, grads)
    
    # Check that parameters have changed
    @model.parameters.each do |name, param|
      refute_array_equal(param, old_params[name])
    end
    
    # Check that optimizer state contains expected keys
    assert optimizer.state.key?("count")
    assert optimizer.state.key?("momentum")
  end
  
  def test_learning_rate_scheduler
    # Create optimizer with scheduler
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
    
    # Add step decay scheduler
    scheduler = MLX::Optimizers::StepLR.new(
      optimizer, 
      step_size: 5, 
      gamma: 0.5
    )
    
    # Initial learning rate
    assert_equal 0.1, optimizer.learning_rate
    
    # Simulate 10 updates
    10.times do |i|
      # Compute loss and gradients
      loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
      
      # Update parameters
      optimizer.update(@model, grads)
      
      # Step the scheduler
      scheduler.step
      
      # Check learning rate
      expected_lr = 0.1 * (0.5 ** (i // 5))
      assert_in_delta expected_lr, optimizer.learning_rate, 1e-6
    end
  end
  
  def test_cosine_annealing_scheduler
    # Skip if CosineAnnealingLR is not implemented
    skip unless defined?(MLX::Optimizers::CosineAnnealingLR)
    
    # Create optimizer with scheduler
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
    
    # Add cosine annealing scheduler
    scheduler = MLX::Optimizers::CosineAnnealingLR.new(
      optimizer, 
      T_max: 10,
      eta_min: 0.001
    )
    
    # Initial learning rate
    assert_equal 0.1, optimizer.learning_rate
    
    # Simulate 10 updates
    10.times do |i|
      # Compute loss and gradients
      loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
      
      # Update parameters
      optimizer.update(@model, grads)
      
      # Step the scheduler
      scheduler.step
      
      # Learning rate should change according to cosine schedule
      assert optimizer.learning_rate <= 0.1
      assert optimizer.learning_rate >= 0.001
    end
    
    # After T_max steps, learning rate should be at eta_min
    assert_in_delta 0.001, optimizer.learning_rate, 1e-6
  end
  
  def test_exponential_lr_scheduler
    # Skip if ExponentialLR is not implemented
    skip unless defined?(MLX::Optimizers::ExponentialLR)
    
    # Create optimizer with scheduler
    optimizer = MLX::Optimizers::SGD.new(learning_rate: 0.1)
    
    # Add exponential scheduler
    scheduler = MLX::Optimizers::ExponentialLR.new(
      optimizer, 
      gamma: 0.9
    )
    
    # Initial learning rate
    assert_equal 0.1, optimizer.learning_rate
    
    # Simulate 5 updates
    5.times do |i|
      # Compute loss and gradients
      loss, grads = MLX.value_and_grad(@model, @loss_fn).call(@model, @x, @y)
      
      # Update parameters
      optimizer.update(@model, grads)
      
      # Step the scheduler
      scheduler.step
      
      # Check learning rate
      expected_lr = 0.1 * (0.9 ** (i + 1))
      assert_in_delta expected_lr, optimizer.learning_rate, 1e-6
    end
  end
end 