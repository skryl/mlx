module MLX
  module Optimizers
    # Helper method for exponential decay scheduler
    def self.exponential_decay(init, decay_rate)
      -> (step) { init * (decay_rate ** step) }
    end
    
    # Helper method for step decay scheduler
    def self.step_decay(init, decay_rate, step_size)
      -> (step) { init * (decay_rate ** (step / step_size)) }
    end
    
    # Helper method for cosine decay scheduler
    def self.cosine_decay(init, decay_steps, final = 0.0)
      -> (step) {
        s = [step, decay_steps].min
        decay = 0.5 * (1.0 + Ops.cos((MLX.pi / decay_steps) * s))
        final + decay * (init - final)
      }
    end
    
    # Helper method for joining schedulers
    def self.join_schedules(schedules, boundaries)
      if schedules.empty?
        raise ArgumentError, "Must provide at least 1 schedule to join"
      end
      
      if schedules.length != boundaries.length + 1
        raise ArgumentError, "Received #{boundaries.length} boundaries but expected #{schedules.length - 1}"
      end
      
      -> (step) {
        output = schedules[0].call(step)
        
        boundaries.each_with_index do |boundary, idx|
          if step >= boundary
            output = schedules[idx + 1].call(step - boundary)
          end
        end
        
        output
      }
    end
    
    # Helper method for linear schedule
    def self.linear_schedule(init, final, steps)
      if steps < 1
        raise ArgumentError, "steps must be greater than 0, but got #{steps}"
      end
      
      -> (step) {
        s = [step, steps].min
        s * ((final - init) / steps) + init
      }
    end
    
    # Base class for learning rate schedulers
    class BaseLRScheduler
      attr_reader :optimizer, :last_epoch, :base_lrs
      
      def initialize(optimizer, last_epoch: -1, verbose: false)
        # Validate optimizer
        unless optimizer.is_a?(MLX::Optimizers::Optimizer)
          raise ArgumentError, "Expected optimizer to be an instance of MLX::Optimizers::Optimizer"
        end
        
        @optimizer = optimizer
        @verbose = verbose
        @last_epoch = last_epoch
        
        # Initialize learning rates
        @base_lrs = []
        optimizer.param_groups.each do |group|
          @base_lrs << group[:lr]
        end
        
        # If last_epoch == -1, initialize as if a first step was done
        if last_epoch == -1
          step
        end
      end
      
      # Step method to update learning rates
      def step(epoch = nil)
        # Set epoch for scheduling
        if epoch.nil?
          @last_epoch += 1
          epoch = @last_epoch
        else
          @last_epoch = epoch
        end
        
        # Get new learning rates
        values = get_lr(epoch)
        
        # Apply new learning rates to optimizer
        @optimizer.param_groups.each_with_index do |group, i|
          group[:lr] = values[i]
          
          # Log change if verbose
          if @verbose
            puts "Adjusting learning rate of group #{i} to #{values[i]}"
          end
        end
      end
      
      # Method to be implemented by subclasses to calculate learning rates
      def get_lr(epoch)
        raise NotImplementedError, "Subclass must implement get_lr"
      end
    end
    
    # Step decay learning rate scheduler
    class StepLR < BaseLRScheduler
      attr_reader :step_size, :gamma
      
      def initialize(optimizer, step_size, gamma: 0.1, last_epoch: -1, verbose: false)
        @step_size = step_size
        @gamma = gamma
        
        super(optimizer, last_epoch: last_epoch, verbose: verbose)
      end
      
      # Calculate learning rates based on step decay
      def get_lr(epoch)
        # Apply decay for each step_size
        decay_factor = @gamma ** (epoch / @step_size)
        
        # Apply decay to all base learning rates
        @base_lrs.map { |lr| lr * decay_factor }
      end
    end
    
    # Multi-step decay learning rate scheduler
    class MultiStepLR < BaseLRScheduler
      attr_reader :milestones, :gamma
      
      def initialize(optimizer, milestones, gamma: 0.1, last_epoch: -1, verbose: false)
        @milestones = milestones.sort
        @gamma = gamma
        
        super(optimizer, last_epoch: last_epoch, verbose: verbose)
      end
      
      # Calculate learning rates based on milestone decay
      def get_lr(epoch)
        # Count how many milestones have been passed
        decay_count = @milestones.count { |m| m <= epoch }
        decay_factor = @gamma ** decay_count
        
        # Apply decay to all base learning rates
        @base_lrs.map { |lr| lr * decay_factor }
      end
    end
    
    # Exponential decay learning rate scheduler
    class ExponentialLR < BaseLRScheduler
      attr_reader :gamma
      
      def initialize(optimizer, gamma, last_epoch: -1, verbose: false)
        @gamma = gamma
        
        super(optimizer, last_epoch: last_epoch, verbose: verbose)
      end
      
      # Calculate learning rates based on exponential decay
      def get_lr(epoch)
        # Apply exponential decay
        decay_factor = @gamma ** epoch
        
        # Apply decay to all base learning rates
        @base_lrs.map { |lr| lr * decay_factor }
      end
    end
    
    # Cosine annealing learning rate scheduler
    class CosineAnnealingLR < BaseLRScheduler
      attr_reader :t_max, :eta_min
      
      def initialize(optimizer, t_max, eta_min: 0, last_epoch: -1, verbose: false)
        @t_max = t_max
        @eta_min = eta_min
        
        super(optimizer, last_epoch: last_epoch, verbose: verbose)
      end
      
      # Calculate learning rates based on cosine annealing
      def get_lr(epoch)
        # Handle special case when epoch >= t_max
        if epoch >= @t_max
          return @base_lrs.map { |_| @eta_min }
        end
        
        # Calculate cosine decay factor
        cos_factor = (1 + Ops.cos(MLX.pi * epoch / @t_max)) / 2.0
        
        # Apply cosine annealing to all base learning rates
        @base_lrs.map do |base_lr|
          @eta_min + (base_lr - @eta_min) * cos_factor
        end
      end
    end
    
    # Reduce learning rate on plateau scheduler
    class ReduceLROnPlateau
      attr_reader :optimizer, :mode, :factor, :patience, :threshold, 
                  :threshold_mode, :cooldown, :min_lr, :eps, :verbose
      
      def initialize(optimizer, mode: 'min', factor: 0.1, patience: 10, 
                    threshold: 1e-4, threshold_mode: 'rel', cooldown: 0, 
                    min_lr: 0, eps: 1e-8, verbose: false)
        # Validate optimizer
        unless optimizer.is_a?(MLX::Optimizers::Optimizer)
          raise ArgumentError, "Expected optimizer to be an instance of MLX::Optimizers::Optimizer"
        end
        
        # Validate mode
        unless ['min', 'max'].include?(mode)
          raise ArgumentError, "Mode #{mode} is unknown, expected one of ('min', 'max')"
        end
        
        # Validate threshold mode
        unless ['rel', 'abs'].include?(threshold_mode)
          raise ArgumentError, "Threshold mode #{threshold_mode} is unknown, expected one of ('rel', 'abs')"
        end
        
        # Initialize parameters
        @optimizer = optimizer
        @mode = mode
        @factor = factor
        @patience = patience
        @threshold = threshold
        @threshold_mode = threshold_mode
        @cooldown = cooldown
        @min_lr = min_lr.is_a?(Array) ? min_lr : [min_lr] * optimizer.param_groups.length
        @eps = eps
        @verbose = verbose
        
        # Initialize state
        @best = mode == 'min' ? Float::INFINITY : -Float::INFINITY
        @num_bad_epochs = 0
        @cooldown_counter = 0
        @in_cooldown = false
      end
      
      # Step method to update learning rates based on metric
      def step(metrics)
        # Check if metrics improved
        current = metrics
        
        # Convert metrics to numeric value if needed
        if current.is_a?(Array) || current.is_a?(MLX::Array)
          current = current.first
        end
        
        # Check for improvement
        if @mode == 'min'
          is_better = is_better_than(@best, current)
        else
          is_better = is_better_than(current, @best)
        end
        
        # Update best metric if improved
        if is_better
          @best = current
          @num_bad_epochs = 0
        else
          @num_bad_epochs += 1
        end
        
        # Check if we should reduce learning rate
        if @cooldown_counter > 0
          @cooldown_counter -= 1
          @num_bad_epochs = 0
        end
        
        # If patience exceeded, reduce learning rate
        if @num_bad_epochs > @patience && @cooldown_counter == 0
          # Reduce learning rate for all parameter groups
          @optimizer.param_groups.each_with_index do |group, i|
            old_lr = group[:lr]
            new_lr = [old_lr * @factor, @min_lr[i]].max
            group[:lr] = new_lr
            
            # Log change if verbose
            if @verbose
              puts "Reducing learning rate of group #{i} to #{new_lr}"
            end
          end
          
          # Reset bad epochs counter and start cooldown
          @cooldown_counter = @cooldown
          @num_bad_epochs = 0
        end
      end
      
      private
      
      # Helper method to check if a value is better than another
      def is_better_than(current, best)
        if @threshold_mode == 'rel'
          if @mode == 'min'
            improvement = 1 - current / best
            return improvement > @threshold
          else
            improvement = current / best - 1
            return improvement > @threshold
          end
        else
          if @mode == 'min'
            return best - current > @threshold
          else
            return current - best > @threshold
          end
        end
      end
    end
    
    # Cyclic learning rate scheduler
    class CyclicLR < BaseLRScheduler
      attr_reader :base_lr, :max_lr, :step_size_up, :step_size_down,
                  :mode, :gamma, :scale_fn, :scale_mode, :cycle_momentum,
                  :base_momentum, :max_momentum, :last_epoch
      
      def initialize(optimizer, base_lr, max_lr, step_size_up: 2000, step_size_down: nil, 
                     mode: 'triangular', gamma: 1.0, scale_fn: nil, scale_mode: 'cycle', 
                     cycle_momentum: true, base_momentum: 0.8, max_momentum: 0.9, 
                     last_epoch: -1, verbose: false)
        # Validate inputs
        if !['triangular', 'triangular2', 'exp_range'].include?(mode) && scale_fn.nil?
          raise ArgumentError, "Mode #{mode} not recognized. Must be one of 'triangular', 'triangular2', 'exp_range', or a custom scale_fn must be specified"
        end
        
        # Set step_size_down if not provided
        @step_size_up = step_size_up
        @step_size_down = step_size_down || step_size_up
        
        # Calculate total cycle size
        @total_size = @step_size_up + @step_size_down
        
        # Set base_lr and max_lr
        @base_lr = [base_lr].flatten
        @max_lr = [max_lr].flatten
        
        # Handle case with different lengths
        if @base_lr.length != @max_lr.length
          if @base_lr.length == 1
            @base_lr = [@base_lr[0]] * @max_lr.length
          elsif @max_lr.length == 1
            @max_lr = [@max_lr[0]] * @base_lr.length
          else
            raise ArgumentError, "base_lr and max_lr have different lengths: #{@base_lr.length} and #{@max_lr.length}"
          end
        end
        
        # Set up scaling function based on mode
        @mode = mode
        @gamma = gamma
        @scale_fn = scale_fn
        @scale_mode = scale_mode
        
        if @scale_fn.nil?
          case mode
          when 'triangular'
            @scale_fn = -> (x) { 1.0 }
            @scale_mode = 'cycle'
          when 'triangular2'
            @scale_fn = -> (x) { 1.0 / (2.0 ** (x - 1)) }
            @scale_mode = 'cycle'
          when 'exp_range'
            @scale_fn = -> (x) { gamma ** x }
            @scale_mode = 'iterations'
          end
        end
        
        # Set up momentum parameters
        @cycle_momentum = cycle_momentum
        @base_momentum = [base_momentum].flatten
        @max_momentum = [max_momentum].flatten
        
        # Handle momentum arrays with different lengths
        if @cycle_momentum
          if @base_momentum.length != @max_momentum.length
            if @base_momentum.length == 1
              @base_momentum = [@base_momentum[0]] * @max_momentum.length
            elsif @max_momentum.length == 1
              @max_momentum = [@max_momentum[0]] * @base_momentum.length
            else
              raise ArgumentError, "base_momentum and max_momentum have different lengths: #{@base_momentum.length} and #{@max_momentum.length}"
            end
          end
        end
        
        super(optimizer, last_epoch: last_epoch, verbose: verbose)
      end
      
      # Calculate current learning rate
      def get_lr(epoch)
        # Calculate where in the cycle we are
        cycle = (epoch / @total_size).floor
        
        # Calculate position within the cycle
        x = epoch % @total_size
        
        # Calculate scaling factor
        if x <= @step_size_up
          # Increasing phase
          scale_factor = x / @step_size_up
        else
          # Decreasing phase
          scale_factor = 1.0 - (x - @step_size_up) / @step_size_down
        end
        
        # Apply scaling function
        if @scale_mode == 'cycle'
          scale = @scale_fn.call(cycle)
        else
          scale = @scale_fn.call(epoch)
        end
        
        # Calculate new learning rates
        new_lr = @base_lrs.map.with_index do |base_lr, i|
          lr_diff = @max_lr[i] - base_lr
          base_lr + lr_diff * scale_factor * scale
        end
        
        # Update momentum if needed
        if @cycle_momentum
          @optimizer.param_groups.each_with_index do |group, i|
            if group.key?(:momentum)
              mom_diff = @max_momentum[i] - @base_momentum[i]
              group[:momentum] = @base_momentum[i] + mom_diff * (1 - scale_factor) * scale
            end
          end
        end
        
        new_lr
      end
    end
    
    # One cycle learning rate scheduler
    class OneCycleLR < BaseLRScheduler
      attr_reader :max_lr, :total_steps, :div_factor, :pct_start, :anneal_strategy, 
                  :cycle_momentum, :base_momentum, :max_momentum, :three_phase
      
      def initialize(optimizer, max_lr, total_steps: nil, epochs: nil, steps_per_epoch: nil,
                     pct_start: 0.3, anneal_strategy: 'cos', div_factor: 25.0, final_div_factor: 1e4,
                     three_phase: false, cycle_momentum: true, base_momentum: 0.85, max_momentum: 0.95,
                     last_epoch: -1, verbose: false)
        # Validate input parameters
        unless (total_steps.nil? ^ epochs.nil? ^ steps_per_epoch.nil?)
          raise ArgumentError, "One of total_steps OR (epochs and steps_per_epoch) must be specified"
        end
        
        # Calculate total steps if not directly provided
        if total_steps.nil?
          if epochs.nil? || steps_per_epoch.nil?
            raise ArgumentError, "Both epochs and steps_per_epoch must be specified if total_steps is not given"
          end
          @total_steps = epochs * steps_per_epoch
        else
          @total_steps = total_steps
        end
        
        # Validate other parameters
        unless ['cos', 'linear'].include?(anneal_strategy)
          raise ArgumentError, "anneal_strategy must be one of 'cos' or 'linear', got: #{anneal_strategy}"
        end
        
        if pct_start < 0 || pct_start > 1
          raise ArgumentError, "pct_start must be between 0 and 1, got: #{pct_start}"
        end
        
        # Set up learning rate parameters
        @max_lr = max_lr.is_a?(Array) ? max_lr : [max_lr] * optimizer.param_groups.length
        @div_factor = div_factor
        @final_div_factor = final_div_factor
        @pct_start = pct_start
        @anneal_strategy = anneal_strategy
        @three_phase = three_phase
        
        # Calculate step sizes for each phase
        @step_size_up = (@total_steps * @pct_start).ceil
        
        if @three_phase
          @step_size_down = (@total_steps * @pct_start).ceil
          @step_size_final = @total_steps - @step_size_up - @step_size_down
        else
          @step_size_down = @total_steps - @step_size_up
        end
        
        # Set up momentum parameters
        @cycle_momentum = cycle_momentum
        
        if @cycle_momentum
          @base_momentum = base_momentum.is_a?(Array) ? base_momentum : [base_momentum] * optimizer.param_groups.length
          @max_momentum = max_momentum.is_a?(Array) ? max_momentum : [max_momentum] * optimizer.param_groups.length
          
          # Check if lengths match
          if @base_momentum.length != @max_momentum.length
            raise ArgumentError, "base_momentum and max_momentum have different lengths: #{@base_momentum.length} and #{@max_momentum.length}"
          end
        end
        
        # Initialize base learning rates
        base_lrs = @max_lr.map { |lr| lr / div_factor }
        
        # Store attributes for step function
        @initial_lr = base_lrs.dup
        
        # Set base LRs in parent class
        @optimizer.param_groups.each_with_index do |group, i|
          group[:lr] = base_lrs[i]
        end
        
        super(optimizer, last_epoch: last_epoch, verbose: verbose)
      end
      
      # Calculate learning rates for current step
      def get_lr(epoch)
        if epoch <= @step_size_up
          # Increasing phase
          if @anneal_strategy == 'cos'
            # Cosine annealing
            phase_ratio = epoch / @step_size_up
            cos_factor = (1 - Ops.cos(phase_ratio * MLX.pi)) / 2
            @base_lrs.map.with_index do |base_lr, i|
              base_lr + (@max_lr[i] - base_lr) * cos_factor
            end
          else
            # Linear annealing
            phase_ratio = epoch / @step_size_up
            @base_lrs.map.with_index do |base_lr, i|
              base_lr + (@max_lr[i] - base_lr) * phase_ratio
            end
          end
        elsif @three_phase && epoch <= @step_size_up + @step_size_down
          # Middle phase (three_phase only)
          if @anneal_strategy == 'cos'
            # Cosine annealing
            phase_ratio = (epoch - @step_size_up) / @step_size_down
            cos_factor = (1 + Ops.cos(phase_ratio * MLX.pi)) / 2
            @max_lr.map.with_index do |max_lr, i|
              max_lr + (base_lr - max_lr) * cos_factor
            end
          else
            # Linear annealing
            phase_ratio = (epoch - @step_size_up) / @step_size_down
            @max_lr.map.with_index do |max_lr, i|
              max_lr - (max_lr - base_lr) * phase_ratio
            end
          end
        else
          # Decreasing phase
          if @three_phase
            phase = epoch - @step_size_up - @step_size_down
            phase_ratio = phase / @step_size_final
          else
            phase = epoch - @step_size_up
            phase_ratio = phase / @step_size_down
          end
          
          if @anneal_strategy == 'cos'
            # Cosine annealing
            cos_factor = (1 + Ops.cos(phase_ratio * MLX.pi)) / 2
            @base_lrs.map.with_index do |base_lr, i|
              final_lr = base_lr / @final_div_factor
              final_lr + (base_lr - final_lr) * cos_factor
            end
          else
            # Linear annealing
            @base_lrs.map.with_index do |base_lr, i|
              final_lr = base_lr / @final_div_factor
              base_lr - (base_lr - final_lr) * phase_ratio
            end
          end
        end
      end
      
      # Handle momentum updates
      def step(epoch = nil)
        super(epoch)
        
        # Update momentum if needed
        if @cycle_momentum
          if epoch <= @step_size_up
            # Increasing phase
            if @anneal_strategy == 'cos'
              # Cosine annealing
              phase_ratio = epoch / @step_size_up
              cos_factor = (1 - Ops.cos(phase_ratio * MLX.pi)) / 2
              @optimizer.param_groups.each_with_index do |group, i|
                group[:momentum] = @max_momentum[i] - (@max_momentum[i] - @base_momentum[i]) * cos_factor
              end
            else
              # Linear annealing
              phase_ratio = epoch / @step_size_up
              @optimizer.param_groups.each_with_index do |group, i|
                group[:momentum] = @max_momentum[i] - (@max_momentum[i] - @base_momentum[i]) * phase_ratio
              end
            end
          elsif @three_phase && epoch <= @step_size_up + @step_size_down
            # Middle phase (three_phase only)
            if @anneal_strategy == 'cos'
              # Cosine annealing
              phase_ratio = (epoch - @step_size_up) / @step_size_down
              cos_factor = (1 + Ops.cos(phase_ratio * MLX.pi)) / 2
              @optimizer.param_groups.each_with_index do |group, i|
                group[:momentum] = @base_momentum[i] - (@base_momentum[i] - @max_momentum[i]) * cos_factor
              end
            else
              # Linear annealing
              phase_ratio = (epoch - @step_size_up) / @step_size_down
              @optimizer.param_groups.each_with_index do |group, i|
                group[:momentum] = @base_momentum[i] + (@max_momentum[i] - @base_momentum[i]) * phase_ratio
              end
            end
          else
            # Decreasing phase
            if @three_phase
              phase = epoch - @step_size_up - @step_size_down
              phase_ratio = phase / @step_size_final
            else
              phase = epoch - @step_size_up
              phase_ratio = phase / @step_size_down
            end
            
            if @anneal_strategy == 'cos'
              # Cosine annealing
              cos_factor = (1 + Ops.cos(phase_ratio * MLX.pi)) / 2
              @optimizer.param_groups.each_with_index do |group, i|
                group[:momentum] = @base_momentum[i] - (@base_momentum[i] - @max_momentum[i]) * cos_factor
              end
            else
              # Linear annealing
              @optimizer.param_groups.each_with_index do |group, i|
                group[:momentum] = @base_momentum[i] + (@max_momentum[i] - @base_momentum[i]) * phase_ratio
              end
            end
          end
        end
      end
    end
  end
end 