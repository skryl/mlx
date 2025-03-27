module MLX
  module NN
    module Layers
      # Base class for convolution layers
      class BaseConvNd < MLX::NN::Module
        attr_reader :in_channels, :out_channels, :kernel_size, :stride, :padding, :dilation, :groups, :bias, :padding_mode

        def initialize(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
          super()
          @in_channels = in_channels
          @out_channels = out_channels
          @kernel_size = kernel_size
          @stride = stride
          @padding = padding
          @dilation = dilation
          @groups = groups
          @bias = bias
          @padding_mode = padding_mode

          # Validate parameters
          if in_channels % groups != 0
            raise ArgumentError, "in_channels (#{in_channels}) must be divisible by groups (#{groups})"
          end

          if out_channels % groups != 0
            raise ArgumentError, "out_channels (#{out_channels}) must be divisible by groups (#{groups})"
          end

          # Initialize weights
          fan_in = (in_channels / groups) * kernel_size.reduce(:*)
          bound = 1.0 / Math.sqrt(fan_in)
          weight_shape = weight_shape()
          weight = MLX::NN::Init.uniform(weight_shape, -bound, bound)
          register_parameter('weight', weight)

          if bias
            bias_shape = [out_channels]
            bias_tensor = MLX::NN::Init.uniform(bias_shape, -bound, bound)
            register_parameter('bias', bias_tensor)
          end
        end

        # Method to be implemented by subclasses
        def weight_shape
          raise NotImplementedError, "Subclass must implement weight_shape"
        end

        # Method to be implemented by subclasses
        def forward(x)
          raise NotImplementedError, "Subclass must implement forward"
        end

        # Reset parameters to their initial values
        def reset_parameters
          fan_in = (@in_channels / @groups) * @kernel_size.reduce(:*)
          bound = 1.0 / Math.sqrt(fan_in)
          
          # Reset weight
          weight_shape = weight_shape()
          weight = MLX::NN::Init.uniform(weight_shape, -bound, bound)
          @_parameters['weight'] = weight
          
          # Reset bias if present
          if @bias
            bias_tensor = MLX::NN::Init.uniform([@out_channels], -bound, bound)
            @_parameters['bias'] = bias_tensor
          end
        end
      end

      # 1D Convolution layer
      class Conv1d < BaseConvNd
        def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0, dilation: 1, groups: 1, bias: true, padding_mode: 'zeros')
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = [stride].flatten
          padding = [padding].flatten
          dilation = [dilation].flatten
          
          # Ensure all parameter arrays have length 1
          kernel_size = [kernel_size[0]]
          stride = [stride[0]]
          padding = [padding[0]]
          dilation = [dilation[0]]
          
          super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        end
        
        def weight_shape
          [@out_channels, (@in_channels / @groups), @kernel_size[0]]
        end
        
        def forward(x)
          # Apply padding if needed
          if @padding_mode == 'zeros'
            if @padding[0] > 0
              # Add padding to the last dimension (width)
              padding_array = [[0, 0], [0, 0], [@padding[0], @padding[0]]]
              x = MLX.pad(x, padding_array)
            end
          else
            raise NotImplementedError, "Padding mode '#{@padding_mode}' is not implemented"
          end
          
          # Apply convolution
          conv_result = MLX.conv1d(x, @_parameters['weight'], 
                                  stride: @stride[0], 
                                  dilation: @dilation[0],
                                  groups: @groups)
          
          # Add bias if present
          if @bias
            # Reshape bias for broadcasting
            bias_shape = [1, @out_channels, 1]
            bias_reshaped = MLX.reshape(@_parameters['bias'], bias_shape)
            
            conv_result = MLX.add(conv_result, bias_reshaped)
          end
          
          conv_result
        end
      end

      # 2D Convolution layer
      class Conv2d < BaseConvNd
        def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0, dilation: 1, groups: 1, bias: true, padding_mode: 'zeros')
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = [stride].flatten
          padding = [padding].flatten
          dilation = [dilation].flatten
          
          # Ensure all parameter arrays have length 2
          kernel_size = kernel_size.length == 1 ? [kernel_size[0], kernel_size[0]] : kernel_size[0..1]
          stride = stride.length == 1 ? [stride[0], stride[0]] : stride[0..1]
          padding = padding.length == 1 ? [padding[0], padding[0]] : padding[0..1]
          dilation = dilation.length == 1 ? [dilation[0], dilation[0]] : dilation[0..1]
          
          super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        end
        
        def weight_shape
          [@out_channels, (@in_channels / @groups), @kernel_size[0], @kernel_size[1]]
        end
        
        def forward(x)
          # Apply padding if needed
          if @padding_mode == 'zeros'
            if @padding[0] > 0 || @padding[1] > 0
              # Add padding to the last two dimensions (height, width)
              padding_array = [[0, 0], [0, 0], [@padding[0], @padding[0]], [@padding[1], @padding[1]]]
              x = MLX.pad(x, padding_array)
            end
          else
            raise NotImplementedError, "Padding mode '#{@padding_mode}' is not implemented"
          end
          
          # Apply convolution
          conv_result = MLX.conv2d(x, @_parameters['weight'], 
                                  stride: @stride, 
                                  dilation: @dilation,
                                  groups: @groups)
          
          # Add bias if present
          if @bias
            # Reshape bias for broadcasting
            bias_shape = [1, @out_channels, 1, 1]
            bias_reshaped = MLX.reshape(@_parameters['bias'], bias_shape)
            
            conv_result = MLX.add(conv_result, bias_reshaped)
          end
          
          conv_result
        end
      end

      # 3D Convolution layer
      class Conv3d < BaseConvNd
        def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0, dilation: 1, groups: 1, bias: true, padding_mode: 'zeros')
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = [stride].flatten
          padding = [padding].flatten
          dilation = [dilation].flatten
          
          # Ensure all parameter arrays have length 3
          kernel_size = case kernel_size.length
                        when 1 then [kernel_size[0], kernel_size[0], kernel_size[0]]
                        when 2 then [kernel_size[0], kernel_size[1], kernel_size[1]]
                        else kernel_size[0..2]
                        end
                        
          stride = case stride.length
                  when 1 then [stride[0], stride[0], stride[0]]
                  when 2 then [stride[0], stride[1], stride[1]]
                  else stride[0..2]
                  end
                  
          padding = case padding.length
                    when 1 then [padding[0], padding[0], padding[0]]
                    when 2 then [padding[0], padding[1], padding[1]]
                    else padding[0..2]
                    end
                    
          dilation = case dilation.length
                    when 1 then [dilation[0], dilation[0], dilation[0]]
                    when 2 then [dilation[0], dilation[1], dilation[1]]
                    else dilation[0..2]
                    end
          
          super(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        end
        
        def weight_shape
          [@out_channels, (@in_channels / @groups), @kernel_size[0], @kernel_size[1], @kernel_size[2]]
        end
        
        def forward(x)
          # Apply padding if needed
          if @padding_mode == 'zeros'
            if @padding.any? { |p| p > 0 }
              # Add padding to the last three dimensions (depth, height, width)
              padding_array = [
                [0, 0], [0, 0], 
                [@padding[0], @padding[0]], 
                [@padding[1], @padding[1]], 
                [@padding[2], @padding[2]]
              ]
              x = MLX.pad(x, padding_array)
            end
          else
            raise NotImplementedError, "Padding mode '#{@padding_mode}' is not implemented"
          end
          
          # Apply convolution
          conv_result = MLX.conv3d(x, @_parameters['weight'], 
                                  stride: @stride, 
                                  dilation: @dilation,
                                  groups: @groups)
          
          # Add bias if present
          if @bias
            # Reshape bias for broadcasting
            bias_shape = [1, @out_channels, 1, 1, 1]
            bias_reshaped = MLX.reshape(@_parameters['bias'], bias_shape)
            
            conv_result = MLX.add(conv_result, bias_reshaped)
          end
          
          conv_result
        end
      end

      # Transposed convolution layers

      # 1D transposed convolution layer
      class ConvTranspose1d < MLX::NN::Module
        attr_reader :in_channels, :out_channels, :kernel_size, :stride, :padding, :output_padding, :dilation, :groups, :bias

        def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0, output_padding: 0, dilation: 1, groups: 1, bias: true)
          super()
          @in_channels = in_channels
          @out_channels = out_channels
          @kernel_size = kernel_size
          @stride = stride
          @padding = padding
          @output_padding = output_padding
          @dilation = dilation
          @groups = groups
          @bias = bias

          # Calculate weight shape
          if groups != 1
            raise ArgumentError, "ConvTranspose1d currently only supports groups=1"
          end

          # Weight shape: (in_channels, out_channels, kernel_size)
          # Note: in_channels and out_channels are swapped compared to Conv1d
          scale = 1.0 / Math.sqrt(in_channels * kernel_size)
          weight = MLX::Random.uniform(-scale, scale, [in_channels, out_channels, kernel_size])
          register_parameter("weight", weight)

          if bias
            b = MLX::Random.uniform(-scale, scale, [out_channels])
            register_parameter("bias", b)
          end
        end

        def forward(x)
          # Implement transposed convolution using MLX primitives
          result = MLX::Core.conv_transpose1d(x, @weight, stride: @stride, padding: @padding, 
                                       output_padding: @output_padding, dilation: @dilation, groups: @groups)
          
          # Apply bias if present
          result = result + @bias.reshape([1, @out_channels, 1]) if @bias
          
          result
        end
      end

      # 2D transposed convolution layer
      class ConvTranspose2d < MLX::NN::Module
        attr_reader :in_channels, :out_channels, :kernel_size, :stride, :padding, :output_padding, :dilation, :groups, :bias

        def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0, output_padding: 0, dilation: 1, groups: 1, bias: true)
          super()
          @in_channels = in_channels
          @out_channels = out_channels
          @kernel_size = kernel_size.is_a?(Array) ? kernel_size : [kernel_size, kernel_size]
          @stride = stride.is_a?(Array) ? stride : [stride, stride]
          @padding = padding.is_a?(Array) ? padding : [padding, padding]
          @output_padding = output_padding.is_a?(Array) ? output_padding : [output_padding, output_padding]
          @dilation = dilation.is_a?(Array) ? dilation : [dilation, dilation]
          @groups = groups
          @bias = bias

          if groups != 1
            raise ArgumentError, "ConvTranspose2d currently only supports groups=1"
          end

          # Weight shape: (in_channels, out_channels, kernel_h, kernel_w)
          # Note: in_channels and out_channels are swapped compared to Conv2d
          kernel_h, kernel_w = @kernel_size
          scale = 1.0 / Math.sqrt(in_channels * kernel_h * kernel_w)
          weight = MLX::Random.uniform(-scale, scale, [in_channels, out_channels, kernel_h, kernel_w])
          register_parameter("weight", weight)

          if bias
            b = MLX::Random.uniform(-scale, scale, [out_channels])
            register_parameter("bias", b)
          end
        end

        def forward(x)
          # Implement transposed convolution using MLX primitives
          result = MLX::Core.conv_transpose2d(x, @weight, stride: @stride, padding: @padding, 
                                       output_padding: @output_padding, dilation: @dilation, groups: @groups)
          
          # Apply bias if present
          result = result + @bias.reshape([1, @out_channels, 1, 1]) if @bias
          
          result
        end
      end

      # 3D transposed convolution layer
      class ConvTranspose3d < MLX::NN::Module
        attr_reader :in_channels, :out_channels, :kernel_size, :stride, :padding, :output_padding, :dilation, :groups, :bias

        def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0, output_padding: 0, dilation: 1, groups: 1, bias: true)
          super()
          @in_channels = in_channels
          @out_channels = out_channels
          @kernel_size = kernel_size.is_a?(Array) ? kernel_size : [kernel_size, kernel_size, kernel_size]
          @stride = stride.is_a?(Array) ? stride : [stride, stride, stride]
          @padding = padding.is_a?(Array) ? padding : [padding, padding, padding]
          @output_padding = output_padding.is_a?(Array) ? output_padding : [output_padding, output_padding, output_padding]
          @dilation = dilation.is_a?(Array) ? dilation : [dilation, dilation, dilation]
          @groups = groups
          @bias = bias

          if groups != 1
            raise ArgumentError, "ConvTranspose3d currently only supports groups=1"
          end

          # Weight shape: (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
          # Note: in_channels and out_channels are swapped compared to Conv3d
          kernel_d, kernel_h, kernel_w = @kernel_size
          scale = 1.0 / Math.sqrt(in_channels * kernel_d * kernel_h * kernel_w)
          weight = MLX::Random.uniform(-scale, scale, [in_channels, out_channels, kernel_d, kernel_h, kernel_w])
          register_parameter("weight", weight)

          if bias
            b = MLX::Random.uniform(-scale, scale, [out_channels])
            register_parameter("bias", b)
          end
        end

        def forward(x)
          # Implement transposed convolution using MLX primitives
          result = MLX::Core.conv_transpose3d(x, @weight, stride: @stride, padding: @padding, 
                                       output_padding: @output_padding, dilation: @dilation, groups: @groups)
          
          # Apply bias if present
          result = result + @bias.reshape([1, @out_channels, 1, 1, 1]) if @bias
          
          result
        end
      end
    end
  end
end 