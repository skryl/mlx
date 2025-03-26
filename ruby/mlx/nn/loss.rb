module MLX
  module NN
    module Loss
      # Base class for all loss functions
      class _Loss < MLX::NN::Module
        attr_reader :reduction
        
        def initialize(reduction: 'mean')
          super()
          @reduction = reduction
          
          unless ['none', 'mean', 'sum'].include?(reduction)
            raise ArgumentError, "reduction must be 'none', 'mean', or 'sum'"
          end
        end
        
        def forward(input, target)
          unreduced_loss = compute_loss(input, target)
          
          case @reduction
          when 'none'
            unreduced_loss
          when 'mean'
            MLX.mean(unreduced_loss)
          when 'sum'
            MLX.sum(unreduced_loss)
          end
        end
        
        def compute_loss(input, target)
          raise NotImplementedError, "Subclass must implement compute_loss"
        end
        
        alias_method :call, :forward
      end
      
      # Mean Squared Error Loss: (x - y)^2
      class MSELoss < _Loss
        def compute_loss(input, target)
          # Ensure same shape
          if input.shape != target.shape
            raise ArgumentError, "input and target must have the same shape"
          end
          
          # Calculate squared difference
          MLX.square(MLX.subtract(input, target))
        end
      end
      
      # Mean Absolute Error Loss: |x - y|
      class L1Loss < _Loss
        def compute_loss(input, target)
          # Ensure same shape
          if input.shape != target.shape
            raise ArgumentError, "input and target must have the same shape"
          end
          
          # Calculate absolute difference
          MLX.abs(MLX.subtract(input, target))
        end
      end
      
      # Smooth L1 Loss (Huber Loss)
      class SmoothL1Loss < _Loss
        attr_reader :beta
        
        def initialize(beta: 1.0, reduction: 'mean')
          super(reduction: reduction)
          @beta = beta
        end
        
        def compute_loss(input, target)
          # Ensure same shape
          if input.shape != target.shape
            raise ArgumentError, "input and target must have the same shape"
          end
          
          # Calculate difference
          diff = MLX.subtract(input, target)
          abs_diff = MLX.abs(diff)
          
          # Apply smooth L1 formula
          # L = 0.5 * (x)^2 / beta if |x| < beta else |x| - 0.5 * beta
          mask = abs_diff < @beta
          smooth_l1 = MLX.where(
            mask,
            MLX.multiply(0.5 / @beta, MLX.square(abs_diff)),
            MLX.subtract(abs_diff, 0.5 * @beta)
          )
          
          smooth_l1
        end
      end
      
      # Binary Cross Entropy Loss: -[y*log(x) + (1-y)*log(1-x)]
      class BCELoss < _Loss
        attr_reader :weight, :pos_weight
        
        def initialize(weight: nil, pos_weight: nil, reduction: 'mean')
          super(reduction: reduction)
          @weight = weight
          @pos_weight = pos_weight
        end
        
        def compute_loss(input, target)
          # Ensure same shape
          if input.shape != target.shape
            raise ArgumentError, "input and target must have the same shape"
          end
          
          # Clamp input to avoid log(0)
          eps = 1e-12
          input_clamp = MLX.clip(input, eps, 1.0 - eps)
          
          # Binary cross entropy formula
          loss = MLX.negative(
            MLX.add(
              MLX.multiply(target, MLX.log(input_clamp)),
              MLX.multiply(MLX.subtract(1.0, target), MLX.log(MLX.subtract(1.0, input_clamp)))
            )
          )
          
          # Apply weights if provided
          if @weight
            loss = MLX.multiply(loss, @weight)
          end
          
          # Apply positive weights if provided
          if @pos_weight
            # Weighted BCE: -(pos_weight * y * log(x) + (1-y) * log(1-x))
            pos_term = MLX.multiply(MLX.multiply(@pos_weight, target), MLX.log(input_clamp))
            neg_term = MLX.multiply(MLX.subtract(1.0, target), MLX.log(MLX.subtract(1.0, input_clamp)))
            loss = MLX.negative(MLX.add(pos_term, neg_term))
          end
          
          loss
        end
      end
      
      # Binary Cross Entropy with Logits Loss (combines sigmoid and BCE)
      class BCEWithLogitsLoss < _Loss
        attr_reader :weight, :pos_weight
        
        def initialize(weight: nil, pos_weight: nil, reduction: 'mean')
          super(reduction: reduction)
          @weight = weight
          @pos_weight = pos_weight
        end
        
        def compute_loss(input, target)
          # Ensure same shape
          if input.shape != target.shape
            raise ArgumentError, "input and target must have the same shape"
          end
          
          # For numerical stability, use a different formula than naive sigmoid + BCE
          if @pos_weight.nil?
            # Use the stable formula: max(x, 0) - x*z + log(1 + exp(-|x|))
            loss = MLX.add(
              MLX.subtract(
                MLX.maximum(input, 0),
                MLX.multiply(input, target)
              ),
              MLX.log(MLX.add(1.0, MLX.exp(MLX.negative(MLX.abs(input)))))
            )
          else
            # With pos_weight: pos_weight * target * -log(sigmoid(input)) + (1 - target) * -log(1 - sigmoid(input))
            loss = MLX.add(
              MLX.multiply(
                MLX.multiply(@pos_weight, target),
                MLX.maximum(0, MLX.negative(input))
              ),
              MLX.add(
                MLX.multiply(
                  MLX.subtract(1.0, target),
                  MLX.maximum(0, input)
                ),
                MLX.log(MLX.add(1.0, MLX.exp(MLX.negative(MLX.abs(input)))))
              )
            )
          end
          
          # Apply weights if provided
          if @weight
            loss = MLX.multiply(loss, @weight)
          end
          
          loss
        end
      end
      
      # Cross Entropy Loss for classification
      class CrossEntropyLoss < _Loss
        attr_reader :weight, :ignore_index, :label_smoothing
        
        def initialize(weight: nil, ignore_index: -100, label_smoothing: 0.0, reduction: 'mean')
          super(reduction: reduction)
          @weight = weight
          @ignore_index = ignore_index
          @label_smoothing = label_smoothing
        end
        
        def compute_loss(input, target)
          # Input: (batch_size, num_classes, ...)
          # Target: (batch_size, ...) with class indices
          
          # Get dimensions
          batch_size = input.shape[0]
          num_classes = input.shape[1]
          
          # Check if target is in correct shape (class indices)
          if target.shape[0] != batch_size
            raise ArgumentError, "target batch size must match input batch size"
          end
          
          # Apply log softmax to input
          log_probs = MLX::NN::Layers::ActivationFunctions.log_softmax(input, axis: 1)
          
          # Handle 1D targets (batched) or ND targets
          if target.ndim == 1
            # Direct indexing for 1D targets
            loss = MLX.negative(MLX.gather(log_probs, target, axis: 1))
            loss = MLX.reshape(loss, [batch_size])
          else
            # For ND targets, create an indexing array
            # Reshape target to flat indices
            flat_target = MLX.reshape(target, [-1])
            
            # Create mask for valid targets (not ignore_index)
            mask = flat_target != @ignore_index
            valid_targets = MLX.where(mask, flat_target, 0)
            
            # Reshape log_probs to (batch_size, num_classes, -1)
            transposed_shape = [0, 1] + (2...log_probs.ndim).to_a
            log_probs_t = MLX.transpose(log_probs, transposed_shape)
            log_probs_shape = [batch_size * num_classes, -1]
            log_probs_reshaped = MLX.reshape(log_probs_t, log_probs_shape)
            
            # Get the log probability for each valid target
            valid_log_probs = MLX.gather(log_probs_reshaped, valid_targets)
            
            # Create mask for the loss
            loss = MLX.zeros_like(valid_log_probs)
            loss = MLX.where(mask, MLX.negative(valid_log_probs), loss)
            
            # Reshape loss back to target shape
            loss = MLX.reshape(loss, target.shape)
          end
          
          # Apply label smoothing if enabled
          if @label_smoothing > 0.0
            # Calculate smoothed loss
            smooth_loss = MLX.negative(MLX.mean(log_probs, axis: 1))
            
            # Combine with original loss
            loss = MLX.add(
              MLX.multiply(1.0 - @label_smoothing, loss),
              MLX.multiply(@label_smoothing, smooth_loss)
            )
          end
          
          # Apply weights if provided
          if @weight
            # Create a weight tensor based on target indices
            if target.ndim == 1
              target_weights = MLX.gather(@weight, target)
              loss = MLX.multiply(loss, target_weights)
            else
              # For ND targets, apply weights based on valid targets
              target_weights = MLX.gather(@weight, valid_targets)
              target_weights = MLX.reshape(target_weights, target.shape)
              target_weights = MLX.where(mask, target_weights, 0)
              loss = MLX.multiply(loss, target_weights)
            end
          end
          
          # Mask out ignored indices for ND targets
          if target.ndim > 1
            mask = target != @ignore_index
            loss = MLX.where(mask, loss, 0)
          end
          
          loss
        end
      end
      
      # Kullback-Leibler Divergence Loss
      class KLDivLoss < _Loss
        attr_reader :log_target
        
        def initialize(reduction: 'mean', log_target: false)
          super(reduction: reduction)
          @log_target = log_target
        end
        
        def compute_loss(input, target)
          # Ensure same shape
          if input.shape != target.shape
            raise ArgumentError, "input and target must have the same shape"
          end
          
          if @log_target
            # If target is already log probabilities
            loss = MLX.exp(target) * (target - input)
          else
            # Ensure input is log probabilities
            # KL(P||Q) = P * (log(P) - log(Q))
            loss = target * (MLX.log(MLX.clip(target, 1e-10, Float::INFINITY)) - input)
          end
          
          loss
        end
      end
      
      # Negative Log Likelihood Loss
      class NLLLoss < _Loss
        attr_reader :weight, :ignore_index
        
        def initialize(weight: nil, ignore_index: -100, reduction: 'mean')
          super(reduction: reduction)
          @weight = weight
          @ignore_index = ignore_index
        end
        
        def compute_loss(input, target)
          # Input: (batch_size, num_classes, ...)
          # Target: (batch_size, ...) with class indices
          
          # Get dimensions
          batch_size = input.shape[0]
          num_classes = input.shape[1]
          
          # Check if target is in correct shape (class indices)
          if target.shape[0] != batch_size
            raise ArgumentError, "target batch size must match input batch size"
          end
          
          # Handle 1D targets (batched) or ND targets
          if target.ndim == 1
            # Direct indexing for 1D targets
            loss = MLX.negative(MLX.gather(input, target, axis: 1))
            loss = MLX.reshape(loss, [batch_size])
          else
            # For ND targets, create an indexing array
            # Reshape target to flat indices
            flat_target = MLX.reshape(target, [-1])
            
            # Create mask for valid targets (not ignore_index)
            mask = flat_target != @ignore_index
            valid_targets = MLX.where(mask, flat_target, 0)
            
            # Reshape input to (batch_size, num_classes, -1)
            transposed_shape = [0, 1] + (2...input.ndim).to_a
            input_t = MLX.transpose(input, transposed_shape)
            input_shape = [batch_size * num_classes, -1]
            input_reshaped = MLX.reshape(input_t, input_shape)
            
            # Get the negative log probability for each valid target
            valid_log_probs = MLX.gather(input_reshaped, valid_targets)
            
            # Create mask for the loss
            loss = MLX.zeros_like(valid_log_probs)
            loss = MLX.where(mask, MLX.negative(valid_log_probs), loss)
            
            # Reshape loss back to target shape
            loss = MLX.reshape(loss, target.shape)
          end
          
          # Apply weights if provided
          if @weight
            # Create a weight tensor based on target indices
            if target.ndim == 1
              target_weights = MLX.gather(@weight, target)
              loss = MLX.multiply(loss, target_weights)
            else
              # For ND targets, apply weights based on valid targets
              target_weights = MLX.gather(@weight, valid_targets)
              target_weights = MLX.reshape(target_weights, target.shape)
              target_weights = MLX.where(mask, target_weights, 0)
              loss = MLX.multiply(loss, target_weights)
            end
          end
          
          # Mask out ignored indices for ND targets
          if target.ndim > 1
            mask = target != @ignore_index
            loss = MLX.where(mask, loss, 0)
          end
          
          loss
        end
      end
      
      # Hinge Loss (max margin loss)
      class HingeLoss < _Loss
        attr_reader :margin
        
        def initialize(margin: 1.0, reduction: 'mean')
          super(reduction: reduction)
          @margin = margin
        end
        
        def compute_loss(input, target)
          # Ensure target values are -1 or 1
          if ((target != 1) & (target != -1)).any?
            raise ArgumentError, "target must contain values of -1 or 1"
          end
          
          # Calculate hinge loss: max(0, margin - target * input)
          loss = MLX.maximum(0, MLX.subtract(@margin, MLX.multiply(target, input)))
          
          loss
        end
      end
      
      # Cosine Embedding Loss
      class CosineEmbeddingLoss < _Loss
        attr_reader :margin
        
        def initialize(margin: 0.0, reduction: 'mean')
          super(reduction: reduction)
          @margin = margin
        end
        
        def compute_loss(input1, input2, target)
          # Ensure inputs have the same shape
          if input1.shape != input2.shape
            raise ArgumentError, "input shapes must match"
          end
          
          # Compute cosine similarity
          # cos(x, y) = (x · y) / (||x|| * ||y||)
          dot_product = MLX.sum(MLX.multiply(input1, input2), axis: -1)
          norm1 = MLX.sqrt(MLX.sum(MLX.square(input1), axis: -1))
          norm2 = MLX.sqrt(MLX.sum(MLX.square(input2), axis: -1))
          cos_sim = MLX.divide(dot_product, MLX.multiply(norm1, norm2))
          
          # Compute loss based on target
          # If target = 1, loss = 1 - cos(x1, x2)
          # If target = -1, loss = max(0, cos(x1, x2) - margin)
          pos_loss = MLX.subtract(1.0, cos_sim)
          neg_loss = MLX.maximum(0, MLX.subtract(cos_sim, @margin))
          
          # Apply based on target
          mask = target == 1
          loss = MLX.where(mask, pos_loss, neg_loss)
          
          loss
        end
      end
      
      # Triplet Margin Loss
      class TripletMarginLoss < _Loss
        attr_reader :margin, :p, :swap
        
        def initialize(margin: 1.0, p: 2, swap: false, reduction: 'mean')
          super(reduction: reduction)
          @margin = margin
          @p = p
          @swap = swap
        end
        
        def compute_loss(anchor, positive, negative)
          # Calculate distances
          # d(a, p) = ||a - p||_p
          d_ap = MLX.pow(MLX.sum(MLX.pow(MLX.subtract(anchor, positive).abs, @p), axis: -1), 1.0 / @p)
          
          # d(a, n) = ||a - n||_p
          d_an = MLX.pow(MLX.sum(MLX.pow(MLX.subtract(anchor, negative).abs, @p), axis: -1), 1.0 / @p)
          
          if @swap
            # d(p, n) = ||p - n||_p
            d_pn = MLX.pow(MLX.sum(MLX.pow(MLX.subtract(positive, negative).abs, @p), axis: -1), 1.0 / @p)
            
            # d_an = min(d_an, d_pn)
            d_an = MLX.minimum(d_an, d_pn)
          end
          
          # loss = max(0, margin + d_ap - d_an)
          loss = MLX.maximum(0, MLX.add(@margin, MLX.subtract(d_ap, d_an)))
          
          loss
        end
      end
      
      # Margin Ranking Loss
      class MarginRankingLoss < _Loss
        attr_reader :margin
        
        def initialize(margin: 0.0, reduction: 'mean')
          super(reduction: reduction)
          @margin = margin
        end
        
        def compute_loss(input1, input2, target)
          # Ensure inputs have the same shape
          if input1.shape != input2.shape
            raise ArgumentError, "input shapes must match"
          end
          
          # Compute loss: max(0, -target * (input1 - input2) + margin)
          diff = MLX.subtract(input1, input2)
          neg_target_diff = MLX.negative(MLX.multiply(target, diff))
          loss = MLX.maximum(0, MLX.add(neg_target_diff, @margin))
          
          loss
        end
      end
      
      # Focal Loss for imbalanced classification
      class FocalLoss < _Loss
        attr_reader :alpha, :gamma, :reduction
        
        def initialize(alpha: nil, gamma: 2.0, reduction: 'mean')
          super(reduction: reduction)
          @alpha = alpha
          @gamma = gamma
        end
        
        def compute_loss(input, target)
          # Input: (batch_size, num_classes, ...)
          # Target: (batch_size, ...) with class indices
          
          # Apply softmax to input
          logits = MLX::NN::Layers::ActivationFunctions.softmax(input, axis: 1)
          
          # Get dimensions
          batch_size = input.shape[0]
          num_classes = input.shape[1]
          
          # Convert targets to one-hot
          if target.ndim == 1
            # Create one-hot encoding
            target_one_hot = MLX.one_hot(target, num_classes)
          else
            # For ND targets, reshape and create one-hot
            flat_target = MLX.reshape(target, [-1])
            one_hot = MLX.one_hot(flat_target, num_classes)
            target_shape = target.shape + [num_classes]
            target_one_hot = MLX.reshape(one_hot, target_shape)
            
            # Move class dimension to match input shape
            target_one_hot = MLX.transpose(target_one_hot, [0, -1] + (1...target.ndim).to_a)
          end
          
          # Calculate focal loss: -(1-p)^gamma * log(p)
          # Where p is the probability for the target class
          ce = MLX.negative(MLX.sum(MLX.multiply(target_one_hot, MLX.log(MLX.clip(logits, 1e-10, 1.0))), axis: 1))
          
          # Calculate focal weight: (1-p)^gamma
          p_t = MLX.sum(MLX.multiply(target_one_hot, logits), axis: 1)
          focal_weight = MLX.pow(MLX.subtract(1.0, p_t), @gamma)
          
          # Apply focal weight
          loss = MLX.multiply(focal_weight, ce)
          
          # Apply alpha if provided
          if @alpha
            if @alpha.is_a?(Numeric)
              # Single alpha value for positive class
              alpha_t = MLX.multiply(target_one_hot, @alpha) + MLX.multiply(1 - target_one_hot, 1 - @alpha)
              alpha_t = MLX.sum(alpha_t, axis: 1)
              loss = MLX.multiply(alpha_t, loss)
            else
              # Class-wise alpha values
              alpha_t = MLX.sum(MLX.multiply(target_one_hot, @alpha), axis: 1)
              loss = MLX.multiply(alpha_t, loss)
            end
          end
          
          loss
        end
      end
      
      # Multi-label Soft Margin Loss
      class MultiLabelSoftMarginLoss < _Loss
        attr_reader :weight, :reduction
        
        def initialize(weight: nil, reduction: 'mean')
          super(reduction: reduction)
          @weight = weight
        end
        
        def compute_loss(input, target)
          # Ensure inputs have the same shape
          if input.shape != target.shape
            raise ArgumentError, "input and target must have the same shape"
          end
          
          # Compute loss: sum(-target * log(sigmoid(input)) - (1-target) * log(1-sigmoid(input)))
          loss = MLX.negative(
            MLX.add(
              MLX.multiply(target, MLX.log_sigmoid(input)),
              MLX.multiply(MLX.subtract(1.0, target), MLX.log_sigmoid(MLX.negative(input)))
            )
          )
          
          # Apply weights if provided
          if @weight
            loss = MLX.multiply(loss, @weight)
          end
          
          # Sum over the class dimension if not 'none' reduction
          if @reduction != 'none'
            loss = MLX.sum(loss, axis: 1)
          end
          
          loss
        end
      end
    end
  end
end 