require_relative 'mlx_test_case'

class TestOptimizers < MLXTestCase
  def setup
    @params = {
      "w1" => MLX.random.uniform(shape: [3, 2], dtype: MLX.float32),
      "b1" => MLX.zeros([2], dtype: MLX.float32),
      "w2" => MLX.random.uniform(shape: [2, 1], dtype: MLX.float32),
      "b2" => MLX.zeros([1], dtype: MLX.float32)
    }
    
    # Simple squared loss
    @loss_fn = ->(params, x, y) {
      # Simple 2-layer network
      h1 = x.matmul(params["w1"]) + params["b1"]
      h1_relu = MLX.maximum(h1, 0)
      pred = h1_relu.matmul(params["w2"]) + params["b2"]
      loss = MLX.mean((pred - y)**2)
      loss
    }
    
    # Generate sample data
    @x = MLX.random.uniform(shape: [8, 3], dtype: MLX.float32)
    @y = MLX.random.uniform(shape: [8, 1], dtype: MLX.float32)
  end
  
  def test_sgd
    # Test SGD optimizer
    learning_rate = 0.1
    optimizer = MLX.optimizers.SGD(learning_rate: learning_rate)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Basic checks
    assert_kind_of Hash, @params
    assert_equal 4, @params.size
    
    # Test with momentum
    optimizer = MLX.optimizers.SGD(learning_rate: learning_rate, momentum: 0.9)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Test with weight decay
    optimizer = MLX.optimizers.SGD(learning_rate: learning_rate, weight_decay: 0.01)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Test with momentum and weight decay
    optimizer = MLX.optimizers.SGD(learning_rate: learning_rate, momentum: 0.9, weight_decay: 0.01)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
  end
  
  def test_adam
    # Test Adam optimizer
    learning_rate = 0.01
    optimizer = MLX.optimizers.Adam(learning_rate: learning_rate)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Basic checks
    assert_kind_of Hash, @params
    assert_equal 4, @params.size
    
    # Test with custom beta values
    optimizer = MLX.optimizers.Adam(learning_rate: learning_rate, beta1: 0.8, beta2: 0.99)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Test with weight decay
    optimizer = MLX.optimizers.Adam(learning_rate: learning_rate, weight_decay: 0.01)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Test with different eps
    optimizer = MLX.optimizers.Adam(learning_rate: learning_rate, eps: 1e-5)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
  end
  
  def test_rmsprop
    # Test RMSprop optimizer
    learning_rate = 0.01
    optimizer = MLX.optimizers.RMSprop(learning_rate: learning_rate)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Basic checks
    assert_kind_of Hash, @params
    assert_equal 4, @params.size
    
    # Test with custom decay
    optimizer = MLX.optimizers.RMSprop(learning_rate: learning_rate, decay: 0.95)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Test with momentum
    optimizer = MLX.optimizers.RMSprop(learning_rate: learning_rate, momentum: 0.9)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Test with weight decay
    optimizer = MLX.optimizers.RMSprop(learning_rate: learning_rate, weight_decay: 0.01)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
  end
  
  def test_adagrad
    # Test Adagrad optimizer
    learning_rate = 0.1
    optimizer = MLX.optimizers.Adagrad(learning_rate: learning_rate)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Basic checks
    assert_kind_of Hash, @params
    assert_equal 4, @params.size
    
    # Test with initial accumulator value
    optimizer = MLX.optimizers.Adagrad(learning_rate: learning_rate, initial_accumulator_value: 0.1)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Test with eps
    optimizer = MLX.optimizers.Adagrad(learning_rate: learning_rate, eps: 1e-5)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
    
    # Test with weight decay
    optimizer = MLX.optimizers.Adagrad(learning_rate: learning_rate, weight_decay: 0.01)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    3.times do
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state)
    end
  end
  
  def test_with_schedules
    # Test optimizer with learning rate schedule
    init_lr = 0.1
    steps = 100
    
    # Linear schedule
    schedule = MLX.optimizers.linear_schedule(init_value: init_lr, end_value: 0.001, steps: steps)
    optimizer = MLX.optimizers.SGD(learning_rate: schedule)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    10.times do |i|
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state, i)
      
      # Check that learning rate is decreasing
      expected_lr = init_lr - i * (init_lr - 0.001) / steps
      assert_in_epsilon expected_lr, opt_state[:learning_rate].item, 1e-5
    end
    
    # Cosine schedule
    schedule = MLX.optimizers.cosine_schedule(init_value: init_lr, end_value: 0.001, steps: steps)
    optimizer = MLX.optimizers.Adam(learning_rate: schedule)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    10.times do |i|
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state, i)
    end
    
    # Exponential schedule
    schedule = MLX.optimizers.exponential_schedule(init_value: init_lr, end_value: 0.001, steps: steps)
    optimizer = MLX.optimizers.RMSprop(learning_rate: schedule)
    opt_state = optimizer.init(@params)
    
    # Run a few optimization steps
    10.times do |i|
      grads = MLX.grad(@loss_fn).call(@params, @x, @y)
      @params, opt_state = optimizer.update(@params, grads, opt_state, i)
    end
  end
  
  def test_optimizer_state
    # Test that optimizer state is correctly updated
    learning_rate = 0.1
    optimizer = MLX.optimizers.SGD(learning_rate: learning_rate, momentum: 0.9)
    opt_state = optimizer.init(@params)
    
    # Check initial state
    assert_kind_of Hash, opt_state
    assert opt_state.key?(:momentum)
    
    # Run one optimization step
    grads = MLX.grad(@loss_fn).call(@params, @x, @y)
    _, opt_state_new = optimizer.update(@params, grads, opt_state)
    
    # Verify state is updated
    assert_kind_of Hash, opt_state_new
    assert opt_state_new.key?(:momentum)
    
    # Test Adam state
    optimizer = MLX.optimizers.Adam(learning_rate: learning_rate)
    opt_state = optimizer.init(@params)
    
    # Check initial state
    assert_kind_of Hash, opt_state
    assert opt_state.key?(:m)
    assert opt_state.key?(:v)
    
    # Run one optimization step
    grads = MLX.grad(@loss_fn).call(@params, @x, @y)
    _, opt_state_new = optimizer.update(@params, grads, opt_state)
    
    # Verify state is updated
    assert_kind_of Hash, opt_state_new
    assert opt_state_new.key?(:m)
    assert opt_state_new.key?(:v)
  end
end 