module MLX
  module Optimizers
    # Base class for all optimizers
    class Optimizer
      attr_reader :defaults, :state, :param_groups
      
      def initialize(params, defaults = {})
        # Validate parameters
        if params.nil? || params.empty?
          raise ArgumentError, "optimizer got an empty parameter list"
        end
        
        # Initialize state and parameter groups
        @state = {}
        @param_groups = []
        @defaults = defaults
        
        # Add parameter group
        add_param_group(params)
      end
      
      # Add a param group to the optimizer
      def add_param_group(params)
        # Create new parameter group with default options
        param_group = @defaults.dup
        param_group[:params] = params
        
        # Add group
        @param_groups << param_group
      end
      
      # Zero gradients for all parameters
      def zero_grad
        @param_groups.each do |group|
          group[:params].each do |name, param|
            if param.requires_grad && !param.grad.nil?
              param.grad = MLX.zeros_like(param)
            end
          end
        end
      end
      
      # Step method to be implemented by subclasses
      def step
        raise NotImplementedError, "Subclass must implement step"
      end
    end
    
    # SGD optimizer
    class SGD < Optimizer
      def initialize(params, lr: 0.01, momentum: 0, dampening: 0, weight_decay: 0, nesterov: false)
        # Validate parameters
        if momentum < 0
          raise ArgumentError, "Invalid momentum value: #{momentum}"
        end
        
        if weight_decay < 0
          raise ArgumentError, "Invalid weight_decay value: #{weight_decay}"
        end
        
        if nesterov && (momentum <= 0 || dampening != 0)
          raise ArgumentError, "Nesterov momentum requires a momentum and zero dampening"
        end
        
        defaults = {
          lr: lr,
          momentum: momentum,
          dampening: dampening,
          weight_decay: weight_decay,
          nesterov: nesterov
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          weight_decay = group[:weight_decay]
          momentum = group[:momentum]
          dampening = group[:dampening]
          nesterov = group[:nesterov]
          lr = group[:lr]
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            d_p = param.grad
            
            # Apply weight decay
            if weight_decay != 0
              d_p = MLX.add(d_p, MLX.multiply(weight_decay, param))
            end
            
            # Apply momentum
            if momentum != 0
              param_state = @state[name] ||= {}
              
              if param_state[:momentum_buffer].nil?
                # Initialize momentum buffer
                param_state[:momentum_buffer] = d_p.copy()
              else
                # Update momentum buffer
                buf = param_state[:momentum_buffer]
                buf = MLX.add(MLX.multiply(momentum, buf), MLX.multiply(1 - dampening, d_p))
                param_state[:momentum_buffer] = buf
              end
              
              if nesterov
                d_p = MLX.add(d_p, MLX.multiply(momentum, param_state[:momentum_buffer]))
              else
                d_p = param_state[:momentum_buffer]
              end
            end
            
            # Update parameter
            param = MLX.subtract(param, MLX.multiply(lr, d_p))
            group[:params][name] = param
          end
        end
      end
    end
    
    # Adam optimizer
    class Adam < Optimizer
      def initialize(params, lr: 0.001, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0, amsgrad: false)
        # Validate parameters
        if lr <= 0
          raise ArgumentError, "Learning rate must be positive, got #{lr}"
        end
        
        if betas[0] < 0 || betas[0] >= 1
          raise ArgumentError, "Invalid beta1 parameter: #{betas[0]}"
        end
        
        if betas[1] < 0 || betas[1] >= 1
          raise ArgumentError, "Invalid beta2 parameter: #{betas[1]}"
        end
        
        if eps <= 0
          raise ArgumentError, "Invalid epsilon value: #{eps}"
        end
        
        if weight_decay < 0
          raise ArgumentError, "Invalid weight_decay value: #{weight_decay}"
        end
        
        defaults = {
          lr: lr,
          betas: betas,
          eps: eps,
          weight_decay: weight_decay,
          amsgrad: amsgrad
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          lr = group[:lr]
          betas = group[:betas]
          eps = group[:eps]
          weight_decay = group[:weight_decay]
          amsgrad = group[:amsgrad]
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            grad = param.grad
            
            # Get state for this parameter
            param_state = @state[name] ||= {
              step: 0,
              exp_avg: MLX.zeros_like(param),
              exp_avg_sq: MLX.zeros_like(param)
            }
            
            if amsgrad
              param_state[:max_exp_avg_sq] ||= MLX.zeros_like(param)
            end
            
            # Update step count
            param_state[:step] += 1
            step = param_state[:step]
            
            # Get cached values
            exp_avg = param_state[:exp_avg]
            exp_avg_sq = param_state[:exp_avg_sq]
            
            beta1, beta2 = betas
            
            # Apply weight decay
            if weight_decay != 0
              grad = MLX.add(grad, MLX.multiply(weight_decay, param))
            end
            
            # Update biased first moment estimate
            exp_avg = MLX.add(
              MLX.multiply(beta1, exp_avg),
              MLX.multiply(1 - beta1, grad)
            )
            
            # Update biased second raw moment estimate
            exp_avg_sq = MLX.add(
              MLX.multiply(beta2, exp_avg_sq),
              MLX.multiply(1 - beta2, MLX.square(grad))
            )
            
            if amsgrad
              # Maintains the maximum of all 2nd moment running averages
              max_exp_avg_sq = param_state[:max_exp_avg_sq]
              max_exp_avg_sq = MLX.maximum(max_exp_avg_sq, exp_avg_sq)
              param_state[:max_exp_avg_sq] = max_exp_avg_sq
              denom = MLX.add(MLX.sqrt(max_exp_avg_sq), eps)
            else
              denom = MLX.add(MLX.sqrt(exp_avg_sq), eps)
            end
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * Ops.sqrt(bias_correction2) / bias_correction1
            
            # Update parameters
            param = MLX.subtract(
              param,
              MLX.divide(MLX.multiply(step_size, exp_avg), denom)
            )
            
            # Update state
            param_state[:exp_avg] = exp_avg
            param_state[:exp_avg_sq] = exp_avg_sq
            
            # Save parameter
            group[:params][name] = param
          end
        end
      end
    end
    
    # AdamW optimizer (Adam with decoupled weight decay)
    class AdamW < Optimizer
      def initialize(params, lr: 0.001, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0.01, amsgrad: false)
        # Validate parameters
        if lr <= 0
          raise ArgumentError, "Learning rate must be positive, got #{lr}"
        end
        
        if betas[0] < 0 || betas[0] >= 1
          raise ArgumentError, "Invalid beta1 parameter: #{betas[0]}"
        end
        
        if betas[1] < 0 || betas[1] >= 1
          raise ArgumentError, "Invalid beta2 parameter: #{betas[1]}"
        end
        
        if eps <= 0
          raise ArgumentError, "Invalid epsilon value: #{eps}"
        end
        
        if weight_decay < 0
          raise ArgumentError, "Invalid weight_decay value: #{weight_decay}"
        end
        
        defaults = {
          lr: lr,
          betas: betas,
          eps: eps,
          weight_decay: weight_decay,
          amsgrad: amsgrad
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          lr = group[:lr]
          betas = group[:betas]
          eps = group[:eps]
          weight_decay = group[:weight_decay]
          amsgrad = group[:amsgrad]
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            grad = param.grad
            
            # Get state for this parameter
            param_state = @state[name] ||= {
              step: 0,
              exp_avg: MLX.zeros_like(param),
              exp_avg_sq: MLX.zeros_like(param)
            }
            
            if amsgrad
              param_state[:max_exp_avg_sq] ||= MLX.zeros_like(param)
            end
            
            # Update step count
            param_state[:step] += 1
            step = param_state[:step]
            
            # Get cached values
            exp_avg = param_state[:exp_avg]
            exp_avg_sq = param_state[:exp_avg_sq]
            
            beta1, beta2 = betas
            
            # Update biased first moment estimate
            exp_avg = MLX.add(
              MLX.multiply(beta1, exp_avg),
              MLX.multiply(1 - beta1, grad)
            )
            
            # Update biased second raw moment estimate
            exp_avg_sq = MLX.add(
              MLX.multiply(beta2, exp_avg_sq),
              MLX.multiply(1 - beta2, MLX.square(grad))
            )
            
            if amsgrad
              # Maintains the maximum of all 2nd moment running averages
              max_exp_avg_sq = param_state[:max_exp_avg_sq]
              max_exp_avg_sq = MLX.maximum(max_exp_avg_sq, exp_avg_sq)
              param_state[:max_exp_avg_sq] = max_exp_avg_sq
              denom = MLX.add(MLX.sqrt(max_exp_avg_sq), eps)
            else
              denom = MLX.add(MLX.sqrt(exp_avg_sq), eps)
            end
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * Ops.sqrt(bias_correction2) / bias_correction1
            
            # Update parameters
            param_update = MLX.divide(MLX.multiply(step_size, exp_avg), denom)
            
            # Decoupled weight decay
            if weight_decay != 0
              param = MLX.subtract(
                MLX.multiply(param, 1 - lr * weight_decay),
                param_update
              )
            else
              param = MLX.subtract(param, param_update)
            end
            
            # Update state
            param_state[:exp_avg] = exp_avg
            param_state[:exp_avg_sq] = exp_avg_sq
            
            # Save parameter
            group[:params][name] = param
          end
        end
      end
    end
    
    # RMSprop optimizer
    class RMSprop < Optimizer
      def initialize(params, lr: 0.01, alpha: 0.99, eps: 1e-8, weight_decay: 0, momentum: 0, centered: false)
        # Validate parameters
        if lr <= 0
          raise ArgumentError, "Learning rate must be positive, got #{lr}"
        end
        
        if eps <= 0
          raise ArgumentError, "Invalid epsilon value: #{eps}"
        end
        
        defaults = {
          lr: lr,
          alpha: alpha,
          eps: eps,
          weight_decay: weight_decay,
          momentum: momentum,
          centered: centered
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          lr = group[:lr]
          alpha = group[:alpha]
          eps = group[:eps]
          weight_decay = group[:weight_decay]
          momentum = group[:momentum]
          centered = group[:centered]
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            grad = param.grad
            
            # Apply weight decay
            if weight_decay != 0
              grad = MLX.add(grad, MLX.multiply(weight_decay, param))
            end
            
            # Get state for this parameter
            param_state = @state[name] ||= {
              square_avg: MLX.zeros_like(param)
            }
            
            square_avg = param_state[:square_avg]
            
            # Update running average of squared gradients
            square_avg = MLX.add(
              MLX.multiply(alpha, square_avg),
              MLX.multiply(1 - alpha, MLX.square(grad))
            )
            
            param_state[:square_avg] = square_avg
            
            avg = square_avg
            
            if centered
              # Update running average of gradients
              param_state[:grad_avg] ||= MLX.zeros_like(param)
              grad_avg = param_state[:grad_avg]
              grad_avg = MLX.add(
                MLX.multiply(alpha, grad_avg),
                MLX.multiply(1 - alpha, grad)
              )
              param_state[:grad_avg] = grad_avg
              
              # Compute variance: (E[g^2] - (E[g])^2)
              avg = MLX.subtract(avg, MLX.square(grad_avg))
            end
            
            # Compute step size
            step_size = lr / (MLX.sqrt(avg) + eps)
            
            # Apply momentum if needed
            if momentum > 0
              param_state[:momentum_buffer] ||= MLX.zeros_like(param)
              momentum_buffer = param_state[:momentum_buffer]
              momentum_buffer = MLX.add(
                MLX.multiply(momentum, momentum_buffer),
                MLX.multiply(step_size, grad)
              )
              param_state[:momentum_buffer] = momentum_buffer
              param = MLX.subtract(param, momentum_buffer)
            else
              param = MLX.subtract(param, MLX.multiply(step_size, grad))
            end
            
            # Save parameter
            group[:params][name] = param
          end
        end
      end
    end
    
    # Adagrad optimizer
    class Adagrad < Optimizer
      def initialize(params, lr: 0.01, lr_decay: 0, weight_decay: 0, eps: 1e-10)
        # Validate parameters
        if lr <= 0
          raise ArgumentError, "Learning rate must be positive, got #{lr}"
        end
        
        if lr_decay < 0
          raise ArgumentError, "Invalid learning rate decay: #{lr_decay}"
        end
        
        if weight_decay < 0
          raise ArgumentError, "Invalid weight_decay value: #{weight_decay}"
        end
        
        if eps <= 0
          raise ArgumentError, "Invalid epsilon value: #{eps}"
        end
        
        defaults = {
          lr: lr,
          lr_decay: lr_decay,
          weight_decay: weight_decay,
          eps: eps
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          lr = group[:lr]
          lr_decay = group[:lr_decay]
          weight_decay = group[:weight_decay]
          eps = group[:eps]
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            grad = param.grad
            
            # Get state for this parameter
            param_state = @state[name] ||= {
              step: 0,
              sum: MLX.zeros_like(param)
            }
            
            # Update step count
            param_state[:step] += 1
            step = param_state[:step]
            
            # Apply weight decay
            if weight_decay != 0
              grad = MLX.add(grad, MLX.multiply(weight_decay, param))
            end
            
            # Apply learning rate decay
            step_size = lr / (1 + (step - 1) * lr_decay)
            
            # Update sum of squared gradients
            sum = param_state[:sum]
            sum = MLX.add(sum, MLX.square(grad))
            param_state[:sum] = sum
            
            # Compute step
            std = MLX.add(MLX.sqrt(sum), eps)
            param = MLX.subtract(param, MLX.divide(MLX.multiply(step_size, grad), std))
            
            # Save parameter
            group[:params][name] = param
          end
        end
      end
    end
    
    # AdaDelta optimizer
    class AdaDelta < Optimizer
      def initialize(params, lr: 1.0, rho: 0.9, eps: 1e-6, weight_decay: 0)
        # Validate parameters
        if lr <= 0
          raise ArgumentError, "Learning rate must be positive, got #{lr}"
        end
        
        if rho < 0 || rho >= 1
          raise ArgumentError, "Invalid rho value: #{rho}"
        end
        
        if eps <= 0
          raise ArgumentError, "Invalid epsilon value: #{eps}"
        end
        
        if weight_decay < 0
          raise ArgumentError, "Invalid weight_decay value: #{weight_decay}"
        end
        
        defaults = {
          lr: lr,
          rho: rho,
          eps: eps,
          weight_decay: weight_decay
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          lr = group[:lr]
          rho = group[:rho]
          eps = group[:eps]
          weight_decay = group[:weight_decay]
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            grad = param.grad
            
            # Apply weight decay
            if weight_decay != 0
              grad = MLX.add(grad, MLX.multiply(weight_decay, param))
            end
            
            # Get state for this parameter
            param_state = @state[name] ||= {
              square_avg: MLX.zeros_like(param),
              acc_delta: MLX.zeros_like(param)
            }
            
            square_avg = param_state[:square_avg]
            acc_delta = param_state[:acc_delta]
            
            # Update running average of squared gradients
            square_avg = MLX.add(
              MLX.multiply(rho, square_avg),
              MLX.multiply(1 - rho, MLX.square(grad))
            )
            
            # Compute update
            std = MLX.add(MLX.sqrt(square_avg), eps)
            delta = MLX.divide(
              MLX.multiply(MLX.sqrt(MLX.add(acc_delta, eps)), grad),
              std
            )
            
            # Update running average of squared deltas
            acc_delta = MLX.add(
              MLX.multiply(rho, acc_delta),
              MLX.multiply(1 - rho, MLX.square(delta))
            )
            
            # Apply update
            param = MLX.subtract(param, MLX.multiply(lr, delta))
            
            # Update state
            param_state[:square_avg] = square_avg
            param_state[:acc_delta] = acc_delta
            
            # Save parameter
            group[:params][name] = param
          end
        end
      end
    end
    
    # Lion optimizer
    class Lion < Optimizer
      def initialize(params, lr: 1e-4, betas: [0.9, 0.99], weight_decay: 0.0)
        # Validate parameters
        if lr <= 0
          raise ArgumentError, "Learning rate must be positive, got #{lr}"
        end
        
        if betas[0] < 0 || betas[0] >= 1
          raise ArgumentError, "Invalid beta1 parameter: #{betas[0]}"
        end
        
        if betas[1] < 0 || betas[1] >= 1
          raise ArgumentError, "Invalid beta2 parameter: #{betas[1]}"
        end
        
        defaults = {
          lr: lr,
          betas: betas,
          weight_decay: weight_decay
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          lr = group[:lr]
          betas = group[:betas]
          weight_decay = group[:weight_decay]
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            grad = param.grad
            
            # Get state for this parameter
            param_state = @state[name] ||= {
              exp_avg: MLX.zeros_like(param)
            }
            
            # Get previous momentum value
            exp_avg = param_state[:exp_avg]
            
            # Get beta values
            beta1, beta2 = betas
            
            # Update using the Lion update rule
            # Calculate update direction using momentum
            update = MLX.add(
              MLX.multiply(beta1, exp_avg),
              MLX.multiply(1 - beta1, grad)
            )
            
            # Lion sign-based update
            update_sign = MLX.sign(update)
            
            # Update momentum buffer for next iteration
            param_state[:exp_avg] = MLX.add(
              MLX.multiply(beta2, exp_avg),
              MLX.multiply(1 - beta2, grad)
            )
            
            # Apply weight decay before update if required
            if weight_decay != 0
              param = MLX.multiply(param, 1 - lr * weight_decay)
            end
            
            # Apply update
            param = MLX.subtract(param, MLX.multiply(lr, update_sign))
            
            # Save parameter
            group[:params][name] = param
          end
        end
      end
    end
    
    # MultiOptimizer - combines multiple optimizers for different parameter groups
    class MultiOptimizer < Optimizer
      attr_reader :optimizers, :param_groups_map
      
      def initialize(optimizer_specs)
        super({}, {})  # Empty initialization
        @optimizers = []
        @param_groups_map = {}
        
        optimizer_specs.each_with_index do |spec, idx|
          optimizer = spec[:optimizer]
          param_filter = spec[:param_filter]
          @optimizers << optimizer
          @param_groups_map[idx] = {
            optimizer: optimizer,
            param_filter: param_filter
          }
        end
      end
      
      def add_param_group(param_group)
        raise ArgumentError, "MultiOptimizer requires param_group to include a parameter filter"
      end
      
      def load_state_dict(state_dict)
        state_dict["optimizers"].each_with_index do |opt_state, idx|
          @optimizers[idx].load_state_dict(opt_state)
        end
      end
      
      def state_dict
        {
          "optimizers" => @optimizers.map(&:state_dict)
        }
      end
      
      def zero_grad
        @optimizers.each(&:zero_grad)
      end
      
      def step(params = nil, grads = nil)
        params = params || {}
        grads = grads || {}
        
        # Split parameters according to filters
        filtered_params = {}
        filtered_grads = {}
        
        @param_groups_map.each do |idx, group|
          filtered_params[idx] = {}
          filtered_grads[idx] = {}
          filter = group[:param_filter]
          
          params.each do |name, param|
            if filter.call(name, param)
              filtered_params[idx][name] = param
              filtered_grads[idx][name] = grads[name] if grads.key?(name)
            end
          end
        end
        
        # Apply each optimizer to its parameter group
        @param_groups_map.each do |idx, group|
          optimizer = group[:optimizer]
          optimizer.step(filtered_params[idx], filtered_grads[idx])
        end
        
        self
      end
    end
    
    # Adamax optimizer - a variant of Adam based on the infinity norm
    class Adamax < Optimizer
      def initialize(params, lr: 0.002, betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0)
        # Validate parameters
        if lr <= 0
          raise ArgumentError, "Learning rate must be positive, got #{lr}"
        end
        
        if betas[0] < 0 || betas[0] >= 1
          raise ArgumentError, "Invalid beta1 parameter: #{betas[0]}"
        end
        
        if betas[1] < 0 || betas[1] >= 1
          raise ArgumentError, "Invalid beta2 parameter: #{betas[1]}"
        end
        
        if eps <= 0
          raise ArgumentError, "Invalid epsilon value: #{eps}"
        end
        
        if weight_decay < 0
          raise ArgumentError, "Invalid weight_decay value: #{weight_decay}"
        end
        
        defaults = {
          lr: lr,
          betas: betas,
          eps: eps,
          weight_decay: weight_decay
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          lr = group[:lr]
          betas = group[:betas]
          eps = group[:eps]
          weight_decay = group[:weight_decay]
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            grad = param.grad
            
            # Apply weight decay
            if weight_decay != 0
              grad = MLX.add(grad, MLX.multiply(weight_decay, param))
            end
            
            # Get state for this parameter
            param_state = @state[name] ||= {
              step: 0,
              exp_avg: MLX.zeros_like(param),
              exp_inf: MLX.zeros_like(param)
            }
            
            # Update step count
            param_state[:step] += 1
            step = param_state[:step]
            
            # Get cached values
            exp_avg = param_state[:exp_avg]
            exp_inf = param_state[:exp_inf]
            
            beta1, beta2 = betas
            
            # Update biased first moment estimate
            exp_avg = MLX.add(
              MLX.multiply(beta1, exp_avg),
              MLX.multiply(1 - beta1, grad)
            )
            
            # Update the exponentially weighted infinity norm
            norm_buf = MLX.add(
              MLX.multiply(beta2, exp_inf),
              MLX.multiply(1 - beta2, MLX.abs(grad))
            )
            exp_inf = MLX.maximum(norm_buf, MLX.abs(grad))
            
            # Bias correction
            bias_correction = 1 - beta1 ** step
            step_size = lr / bias_correction
            
            # Update parameters
            param = MLX.subtract(
              param,
              MLX.multiply(step_size, MLX.divide(exp_avg, MLX.add(exp_inf, eps)))
            )
            
            # Update state
            param_state[:exp_avg] = exp_avg
            param_state[:exp_inf] = exp_inf
            
            # Save parameter
            group[:params][name] = param
          end
        end
      end
    end
    
    # Adafactor optimizer - memory-efficient variant of Adam
    class Adafactor < Optimizer
      def initialize(params, lr: nil, eps: [1e-30, 1e-3], clip_threshold: 1.0, 
                     decay_rate: -0.8, beta1: nil, weight_decay: 0.0, 
                     scale_parameter: true, relative_step: true, warmup_init: false)
        # Set defaults based on relative_step
        if lr.nil? && relative_step
          lr = 1.0
        elsif lr.nil? && !relative_step
          raise ArgumentError, "lr must be specified when relative_step=false"
        end
        
        # Extract epsilon values
        eps1, eps2 = eps
        
        defaults = {
          lr: lr,
          eps: eps,
          clip_threshold: clip_threshold,
          decay_rate: decay_rate,
          beta1: beta1,
          weight_decay: weight_decay,
          scale_parameter: scale_parameter,
          relative_step: relative_step,
          warmup_init: warmup_init
        }
        
        super(params, defaults)
      end
      
      def step
        @param_groups.each do |group|
          lr = group[:lr]
          eps = group[:eps]
          clip_threshold = group[:clip_threshold]
          decay_rate = group[:decay_rate]
          beta1 = group[:beta1]
          weight_decay = group[:weight_decay]
          scale_parameter = group[:scale_parameter]
          relative_step = group[:relative_step]
          warmup_init = group[:warmup_init]
          
          # Extract epsilon values
          eps1, eps2 = eps
          
          group[:params].each do |name, param|
            next if param.grad.nil?
            
            grad = param.grad
            param_shape = param.shape
            
            # Get state for this parameter
            param_state = @state[name] ||= {
              step: 0
            }
            
            # Update step count
            param_state[:step] += 1
            step = param_state[:step]
            
            # Get factored state if needed
            if param.ndim >= 2
              # For matrices, factor the second moment
              factored = true
              
              # Initialize factored accumulators if not present
              param_state[:exp_avg_sq_row] ||= MLX.zeros([param_shape[0]], dtype: param.dtype)
              param_state[:exp_avg_sq_col] ||= MLX.zeros([param_shape[1]], dtype: param.dtype)
            else
              # For vectors, use regular moment accumulator
              factored = false
              param_state[:exp_avg_sq] ||= MLX.zeros_like(param)
            end
            
            # Initialize first moment if beta1 is set
            if beta1
              param_state[:exp_avg] ||= MLX.zeros_like(param)
            end
            
            # Compute learning rate
            if relative_step
              min_step = warmup_init ? 1e-6 * step : 1e-2
              lr_t = [min_step, 1.0 / Ops.sqrt(step)].min
              lr_t = lr_t * lr
            else
              lr_t = lr
            end
            
            # Apply gradient clipping
            update_scale = 1.0
            if clip_threshold > 0
              # Calculate RMS of gradient
              if factored
                rms = Ops.sqrt(MLX.mean(MLX.square(grad)))
              else
                rms = MLX.core.norm(grad) / Ops.sqrt(grad.size)
              end
              
              # Apply clipping
              if rms > clip_threshold
                update_scale = clip_threshold / rms
              end
            end
            
            # Scale gradient
            grad = MLX.multiply(grad, update_scale)
            
            # Update second moment estimates
            if factored
              # Update row and column factors
              exp_avg_sq_row = param_state[:exp_avg_sq_row]
              exp_avg_sq_col = param_state[:exp_avg_sq_col]
              
              # Compute row and column sums of squared gradients
              r_factor = MLX.mean(MLX.square(grad), axis: 1)
              c_factor = MLX.mean(MLX.square(grad), axis: 0)
              
              # Update with decay rate
              decay = (step + 1) ** decay_rate
              exp_avg_sq_row = MLX.add(
                MLX.multiply(decay, exp_avg_sq_row),
                MLX.multiply(1 - decay, r_factor)
              )
              exp_avg_sq_col = MLX.add(
                MLX.multiply(decay, exp_avg_sq_col),
                MLX.multiply(1 - decay, c_factor)
              )
              
              # Store updated values
              param_state[:exp_avg_sq_row] = exp_avg_sq_row
              param_state[:exp_avg_sq_col] = exp_avg_sq_col
              
              # Compute update
              r_factor = MLX.add(MLX.sqrt(exp_avg_sq_row), eps1)
              c_factor = MLX.add(MLX.sqrt(exp_avg_sq_col), eps1)
              
              # Compute scaling factors for gradient
              r_factor = r_factor.reshape(-1, 1)  # column vector
              c_factor = c_factor.reshape(1, -1)  # row vector
              
              # Compute scaled gradient
              scaled_grad = MLX.divide(grad, MLX.multiply(r_factor, c_factor))
            else
              # Update regular second moment
              exp_avg_sq = param_state[:exp_avg_sq]
              decay = (step + 1) ** decay_rate
              exp_avg_sq = MLX.add(
                MLX.multiply(decay, exp_avg_sq),
                MLX.multiply(1 - decay, MLX.square(grad))
              )
              param_state[:exp_avg_sq] = exp_avg_sq
              
              # Compute scaled gradient
              scaled_grad = MLX.divide(grad, MLX.add(MLX.sqrt(exp_avg_sq), eps1))
            end
            
            # Apply momentum if beta1 is set
            if beta1
              exp_avg = param_state[:exp_avg]
              exp_avg = MLX.add(
                MLX.multiply(beta1, exp_avg),
                MLX.multiply(1 - beta1, scaled_grad)
              )
              param_state[:exp_avg] = exp_avg
              scaled_grad = exp_avg
            end
            
            # Scale parameter if needed
            if scale_parameter
              scale = 1.0
              if param.ndim >= 2
                # For matrices, use the geometric mean of row and column averages
                scale = Ops.sqrt(
                  Ops.sqrt(MLX.mean(exp_avg_sq_row) * MLX.mean(exp_avg_sq_col))
                )
              else
                # For vectors, use RMS
                scale = Ops.sqrt(MLX.mean(exp_avg_sq))
              end
              scale = [scale, eps2].max
              scaled_grad = MLX.divide(scaled_grad, scale)
            end
            
            # Apply weight decay
            if weight_decay != 0
              param = MLX.multiply(param, 1 - lr_t * weight_decay)
            end
            
            # Update parameter
            param = MLX.subtract(param, MLX.multiply(lr_t, scaled_grad))
            
            # Save parameter
            group[:params][name] = param
          end
        end
      end
    end
  end
end 