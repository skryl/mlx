module MLX
  module Utils
    # Apply a function to the leaves of a tree structure
    # 
    # @param fn [Proc] Function to apply to the leaves
    # @param tree [Object] The tree structure (Hash, Array, or leaf value)
    # @param rest [Array] Additional trees to pass corresponding values to fn
    # @param is_leaf [Proc, nil] Optional function to determine if a node is a leaf
    # @return [Object] A new tree with transformed values
    def self.tree_map(fn, tree, *rest, is_leaf: nil)
      # If specified custom is_leaf function says this is a leaf, apply fn
      if is_leaf && is_leaf.call(tree)
        return fn.call(tree, *rest)
      end
      
      # Handle Array or Array-like objects
      if tree.is_a?(Array) || tree.is_a?(Enumerator)
        tree_class = tree.class
        return tree_class.new(
          tree.map.with_index do |child, i|
            child_rest = rest.map { |r| r[i] }
            tree_map(fn, child, *child_rest, is_leaf: is_leaf)
          end
        )
      # Handle Hash objects
      elsif tree.is_a?(Hash)
        return tree.keys.each_with_object({}) do |k, new_tree|
          child_rest = rest.map { |r| r[k] }
          new_tree[k] = tree_map(fn, tree[k], *child_rest, is_leaf: is_leaf)
        end
      # Default: it's a leaf value
      else
        return fn.call(tree, *rest)
      end
    end
    
    # Apply a function to the leaves of a tree structure, with path information
    # 
    # @param fn [Proc] Function taking path and leaf value(s)
    # @param tree [Object] The tree structure (Hash, Array, or leaf value)
    # @param rest [Array] Additional trees to pass corresponding values to fn
    # @param is_leaf [Proc, nil] Optional function to determine if a node is a leaf
    # @param path [String, nil] Current path in the tree traversal
    # @return [Object] A new tree with transformed values
    def self.tree_map_with_path(fn, tree, *rest, is_leaf: nil, path: nil)
      # If specified custom is_leaf function says this is a leaf, apply fn
      if is_leaf && is_leaf.call(tree)
        return fn.call(path, tree, *rest)
      end
      
      # Handle Array or Array-like objects
      if tree.is_a?(Array) || tree.is_a?(Enumerator)
        prefix = path ? "#{path}." : ""
        tree_class = tree.class
        return tree_class.new(
          tree.map.with_index do |child, i|
            child_rest = rest.map { |r| r[i] }
            tree_map_with_path(fn, child, *child_rest, is_leaf: is_leaf, path: "#{prefix}#{i}")
          end
        )
      # Handle Hash objects
      elsif tree.is_a?(Hash)
        prefix = path ? "#{path}." : ""
        return tree.keys.each_with_object({}) do |k, new_tree|
          child_rest = rest.map { |r| r[k] }
          new_tree[k] = tree_map_with_path(fn, tree[k], *child_rest, is_leaf: is_leaf, path: "#{prefix}#{k}")
        end
      # Default: it's a leaf value
      else
        return fn.call(path, tree, *rest)
      end
    end
    
    # Flatten a nested structure into a list of (path, value) pairs
    # 
    # @param tree [Object] The tree structure to flatten
    # @param prefix [String] Prefix for path keys
    # @param is_leaf [Proc, nil] Optional function to determine if a node is a leaf
    # @return [Array<Array>] List of [path, value] pairs
    def self.tree_flatten(tree, prefix: "", is_leaf: nil)
      flat_tree = []
      
      if is_leaf && is_leaf.call(tree)
        return [[prefix[1..], tree]] # Remove leading dot
      end
      
      if tree.is_a?(Array) || tree.is_a?(Enumerator)
        tree.each_with_index do |child, i|
          flat_tree.concat(tree_flatten(child, prefix: "#{prefix}.#{i}", is_leaf: is_leaf))
        end
      elsif tree.is_a?(Hash)
        tree.each do |k, v|
          flat_tree.concat(tree_flatten(v, prefix: "#{prefix}.#{k}", is_leaf: is_leaf))
        end
      else
        # It's a leaf value
        return [[prefix[1..], tree]] # Remove leading dot
      end
      
      flat_tree
    end
    
    # Reconstruct a nested structure from its flattened representation
    # 
    # @param flat_tree [Array<Array>] List of [path, value] pairs
    # @return [Object] Reconstructed nested structure
    def self.tree_unflatten(flat_tree)
      # If there's only a single item with an empty path, just return the value
      if flat_tree.size == 1 && flat_tree[0][0] == ""
        return flat_tree[0][1]
      end
      
      # Determine if the top level is an array or hash
      first_key = flat_tree[0][0].split(".", 2)[0]
      is_array = begin
        Integer(first_key)
        true
      rescue
        false
      end
      
      # Collect children recursively
      children = {}
      flat_tree.each do |key, value|
        current_key, *rest_keys = key.split(".", 2)
        next_key = rest_keys.empty? ? "" : rest_keys[0]
        
        children[current_key] ||= []
        children[current_key] << [next_key, value]
      end
      
      # Build the structure based on type
      if is_array
        # Convert string indices to integers and sort
        indices = children.keys.map { |idx| [Integer(idx), idx] }.sort
        
        # Create array with the right size
        result = []
        indices.each do |i, k|
          # Add nil values to fill gaps if needed
          result.fill(nil, result.size...i) if i >= result.size
          result[i] = tree_unflatten(children[k])
        end
        result
      else
        # Build a hash
        result = {}
        children.each do |k, v|
          result[k] = tree_unflatten(v)
        end
        result
      end
    end
    
    # Reduce a tree to a single value by applying a function to all leaves
    # 
    # @param fn [Proc] Function that takes (accumulator, value) and returns new accumulator
    # @param tree [Object] The tree structure to reduce
    # @param initializer [Object, nil] Initial value for the accumulator
    # @param is_leaf [Proc, nil] Optional function to determine if a node is a leaf
    # @return [Object] The accumulated result
    def self.tree_reduce(fn, tree, initializer = nil, is_leaf: nil)
      if is_leaf && is_leaf.call(tree)
        return initializer.nil? ? tree : fn.call(initializer, tree)
      end
      
      accumulator = initializer
      
      if tree.is_a?(Array) || tree.is_a?(Enumerator)
        tree.each do |item|
          accumulator = tree_reduce(fn, item, accumulator, is_leaf: is_leaf)
        end
      elsif tree.is_a?(Hash)
        tree.each_value do |value|
          accumulator = tree_reduce(fn, value, accumulator, is_leaf: is_leaf)
        end
      else
        # It's a leaf value
        return tree if accumulator.nil?
        return fn.call(accumulator, tree)
      end
      
      accumulator
    end
    
    # Merge two trees together
    # 
    # @param tree_a [Object] First tree
    # @param tree_b [Object] Second tree
    # @param merge_fn [Proc, nil] Optional function to handle leaf conflicts
    # @return [Object] Merged tree
    def self.tree_merge(tree_a, tree_b, merge_fn = nil)
      # If one tree is nil, return the other
      return tree_b if tree_a.nil?
      return tree_a if tree_b.nil?
      
      # If both are arrays, merge by position
      if tree_a.is_a?(Array) && tree_b.is_a?(Array)
        result = tree_a.dup
        tree_b.each_with_index do |v, i|
          if i < result.size
            result[i] = tree_merge(result[i], v, merge_fn)
          else
            result << v
          end
        end
        return result
      end
      
      # If both are hashes, merge by key
      if tree_a.is_a?(Hash) && tree_b.is_a?(Hash)
        result = tree_a.dup
        tree_b.each do |k, v|
          if result.key?(k)
            result[k] = tree_merge(result[k], v, merge_fn)
          else
            result[k] = v
          end
        end
        return result
      end
      
      # If both are leaf values, use merge function or default to tree_b
      if merge_fn
        return merge_fn.call(tree_a, tree_b)
      else
        return tree_b
      end
    end
    
    # Clip the gradient values by norm
    # 
    # @param grads [Hash] Gradients
    # @param max_norm [Float] Maximum allowed norm
    # @param norm_type [Float] Type of norm (1.0, 2.0, etc.)
    # @param error_if_nonfinite [Boolean] Raise error for non-finite values
    # @return [Hash] Clipped gradients
    def self.clip_grad_norm(grads, max_norm, norm_type: 2.0, error_if_nonfinite: false)
      # Calculate total norm across all gradients
      total_norm = 0.0
      
      # Flatten the gradients
      flat_grads = tree_flatten(grads)
      
      # Calculate the norm
      if norm_type == Float::INFINITY
        # Infinity norm is the maximum absolute value
        total_norm = flat_grads.map { |_, g| MLX.max(MLX.abs(g)).item }.max
      elsif norm_type == 1.0
        # L1 norm is the sum of absolute values
        total_norm = flat_grads.sum { |_, g| MLX.sum(MLX.abs(g)).item }
      elsif norm_type == 2.0
        # L2 norm is the square root of sum of squares
        sum_sq = flat_grads.sum { |_, g| MLX.sum(MLX.square(g)).item }
        total_norm = Ops.sqrt(sum_sq)
      else
        # General Lp norm
        sum_powered = flat_grads.sum { |_, g| MLX.sum(MLX.power(MLX.abs(g), norm_type)).item }
        total_norm = sum_powered ** (1.0 / norm_type)
      end
      
      # Check for non-finite values
      if error_if_nonfinite && !total_norm.finite?
        raise ArgumentError, "The total norm of order #{norm_type} for gradients is non-finite"
      end
      
      # If the norm is below the threshold, no need to clip
      return grads if total_norm <= max_norm
      
      # Scale all gradients
      clip_coef = max_norm / (total_norm + 1e-6)
      
      # Apply scaling to gradients
      tree_map(
        lambda { |g| MLX.multiply(g, clip_coef) },
        grads
      )
    end
    
    # Clip the gradient values element-wise
    # 
    # @param grads [Hash] Gradients
    # @param clip_value [Float] Maximum allowed value
    # @return [Hash] Clipped gradients
    def self.clip_grad_value(grads, clip_value)
      tree_map(
        lambda { |g| MLX.clip(g, -clip_value, clip_value) },
        grads
      )
    end
    
    # Vectorize a function over arrays
    # 
    # @param fn [Proc] Function to vectorize
    # @param in_axes [Integer, Array<Integer>] Input axes to vectorize
    # @param out_axes [Integer, Array<Integer>] Output axes
    # @return [Proc] Vectorized function
    def self.vmap(fn, in_axes: 0, out_axes: 0)
      lambda do |*args|
        # Process in_axes to match args length
        axes = in_axes.is_a?(Array) ? in_axes : [in_axes] * args.length
        
        # Transpose inputs to move batch dimension to first position if needed
        inputs = args.map.with_index do |arg, i|
          axis = axes[i]
          # Skip nil axes (non-vectorized arguments)
          next arg if axis.nil?
          
          # Move the specified axis to the front
          if axis != 0
            perm = (0...arg.ndim).to_a
            perm.delete_at(axis)
            perm.unshift(axis)
            MLX.transpose(arg, perm)
          else
            arg
          end
        end
        
        # Get batch size from first vectorized argument
        batch_size = nil
        inputs.each_with_index do |arg, i|
          next if axes[i].nil?
          batch_size = arg.shape[0]
          break
        end
        
        # Apply function to each batch element
        results = (0...batch_size).map do |b|
          # Extract slice for each input
          batch_inputs = inputs.map.with_index do |arg, i|
            next arg if axes[i].nil?
            arg[b]
          end
          
          # Call function with this batch's inputs
          fn.call(*batch_inputs)
        end
        
        # Stack results and transpose if needed
        output = MLX.stack(results, axis: 0)
        
        # Move output batch dimension if needed
        if out_axes != 0
          perm = (0...output.ndim).to_a
          perm.delete_at(0)
          perm.insert(out_axes, 0)
          output = MLX.transpose(output, perm)
        end
        
        output
      end
    end
    
    # Gradient Clipping Utilities

    # Clip gradients by global norm
    # 
    # @param grads [Hash] Gradients to clip
    # @param max_norm [Float] Maximum norm
    # @param norm_type [Float] Type of norm to use (default: 2 for L2 norm)
    # @return [Hash] Clipped gradients
    def self.clip_by_norm(grads, max_norm, norm_type = 2.0)
      if max_norm <= 0
        raise ArgumentError, "max_norm must be positive, got #{max_norm}"
      end
      
      # Calculate the total norm of all gradients
      total_norm = self.global_norm(grads, norm_type)
      
      # If the total norm is already less than max_norm, return grads unchanged
      return grads if total_norm <= max_norm
      
      # Calculate the scaling factor
      scale = max_norm / (total_norm + 1e-6)
      
      # Scale all gradients by this factor
      self.tree_map(->(x) { 
        if x.is_a?(MLX::Array)
          x * scale
        else
          x
        end
      }, grads)
    end

    # Clip gradients by value
    #
    # @param grads [Hash] Gradients to clip
    # @param min_value [Float] Minimum value
    # @param max_value [Float] Maximum value
    # @return [Hash] Clipped gradients
    def self.clip_by_value(grads, min_value, max_value)
      if min_value > max_value
        raise ArgumentError, "min_value must be less than or equal to max_value, got #{min_value} and #{max_value}"
      end
      
      # Clip all gradients to be between min_value and max_value
      self.tree_map(->(x) {
        if x.is_a?(MLX::Array)
          MLX.clip(x, min_value, max_value)
        else
          x
        end
      }, grads)
    end

    # Calculate the global norm of gradients
    #
    # @param grads [Hash] Gradients
    # @param norm_type [Float] Type of norm to use (default: 2 for L2 norm)
    # @return [Float] Global norm
    def self.global_norm(grads, norm_type = 2.0)
      # Calculate the sum of norms across all gradients
      sum_sq = self.tree_reduce(
        ->(acc, x) {
          if x.is_a?(MLX::Array)
            acc + MLX.sum(MLX.abs(x) ** norm_type).item
          else
            acc
          end
        },
        grads,
        0.0
      )
      
      # Return the global norm
      sum_sq ** (1.0 / norm_type)
    end

    # Clip the gradient norm for a model's parameters
    #
    # @param model [MLX::NN::Module] Model
    # @param grads [Hash] Gradients for the model
    # @param max_norm [Float] Maximum norm
    # @param norm_type [Float] Type of norm to use (default: 2 for L2 norm)
    # @return [Hash] Clipped gradients
    def self.clip_grad_norm(model, grads, max_norm, norm_type = 2.0)
      self.clip_by_norm(grads, max_norm, norm_type)
    end

    # Clip the gradient values for a model's parameters
    #
    # @param model [MLX::NN::Module] Model
    # @param grads [Hash] Gradients for the model
    # @param min_value [Float] Minimum value
    # @param max_value [Float] Maximum value
    # @return [Hash] Clipped gradients
    def self.clip_grad_value(model, grads, min_value, max_value)
      self.clip_by_value(grads, min_value, max_value)
    end
  end
end 