module MLX
  module NN
    module Layers
      # Base class for pooling layers
      class BasePoolNd < MLX::NN::Module
        attr_reader :kernel_size, :stride, :padding, :ceil_mode

        def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false)
          super()
          @kernel_size = kernel_size
          @stride = stride.nil? ? kernel_size : stride
          @padding = padding
          @ceil_mode = ceil_mode
        end
      end

      # Max pooling operation in 1D
      class MaxPool1d < BasePoolNd
        def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false)
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = stride.nil? ? nil : [stride].flatten
          padding = [padding].flatten
          
          # Ensure all parameter arrays have length 1
          kernel_size = [kernel_size[0]]
          stride = stride.nil? ? nil : [stride[0]]
          padding = [padding[0]]
          
          super(kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
        end
        
        def forward(x)
          # Apply padding if needed
          if @padding[0] > 0
            # Add padding to the last dimension (width)
            padding_array = [[0, 0], [0, 0], [@padding[0], @padding[0]]]
            x = MLX.pad(x, padding_array, 'constant', -Float::INFINITY)
          end
          
          # Apply max pooling
          MLX.max_pool1d(x, @kernel_size[0], stride: @stride[0], ceil_mode: @ceil_mode)
        end
      end

      # Max pooling operation in 2D
      class MaxPool2d < BasePoolNd
        def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false)
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = stride.nil? ? nil : [stride].flatten
          padding = [padding].flatten
          
          # Ensure all parameter arrays have length 2
          kernel_size = kernel_size.length == 1 ? [kernel_size[0], kernel_size[0]] : kernel_size[0..1]
          stride = if stride.nil?
                    kernel_size.dup
                  else
                    stride.length == 1 ? [stride[0], stride[0]] : stride[0..1]
                  end
          padding = padding.length == 1 ? [padding[0], padding[0]] : padding[0..1]
          
          super(kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
        end
        
        def forward(x)
          # Apply padding if needed
          if @padding[0] > 0 || @padding[1] > 0
            # Add padding to the last two dimensions (height, width)
            padding_array = [[0, 0], [0, 0], [@padding[0], @padding[0]], [@padding[1], @padding[1]]]
            x = MLX.pad(x, padding_array, 'constant', -Float::INFINITY)
          end
          
          # Apply max pooling
          MLX.max_pool2d(x, @kernel_size, stride: @stride, ceil_mode: @ceil_mode)
        end
      end

      # Max pooling operation in 3D
      class MaxPool3d < BasePoolNd
        def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false)
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = stride.nil? ? nil : [stride].flatten
          padding = [padding].flatten
          
          # Ensure all parameter arrays have length 3
          kernel_size = case kernel_size.length
                         when 1 then [kernel_size[0], kernel_size[0], kernel_size[0]]
                         when 2 then [kernel_size[0], kernel_size[1], kernel_size[1]]
                         else kernel_size[0..2]
                         end
                         
          stride = if stride.nil?
                    kernel_size.dup
                  else
                    case stride.length
                    when 1 then [stride[0], stride[0], stride[0]]
                    when 2 then [stride[0], stride[1], stride[1]]
                    else stride[0..2]
                    end
                  end
                  
          padding = case padding.length
                     when 1 then [padding[0], padding[0], padding[0]]
                     when 2 then [padding[0], padding[1], padding[1]]
                     else padding[0..2]
                     end
          
          super(kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
        end
        
        def forward(x)
          # Apply padding if needed
          if @padding.any? { |p| p > 0 }
            # Add padding to the last three dimensions (depth, height, width)
            padding_array = [
              [0, 0], [0, 0],
              [@padding[0], @padding[0]],
              [@padding[1], @padding[1]],
              [@padding[2], @padding[2]]
            ]
            x = MLX.pad(x, padding_array, 'constant', -Float::INFINITY)
          end
          
          # Apply max pooling
          MLX.max_pool3d(x, @kernel_size, stride: @stride, ceil_mode: @ceil_mode)
        end
      end

      # Average pooling operation in 1D
      class AvgPool1d < BasePoolNd
        attr_reader :count_include_pad
        
        def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false, count_include_pad: true)
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = stride.nil? ? nil : [stride].flatten
          padding = [padding].flatten
          
          # Ensure all parameter arrays have length 1
          kernel_size = [kernel_size[0]]
          stride = stride.nil? ? nil : [stride[0]]
          padding = [padding[0]]
          
          super(kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
          @count_include_pad = count_include_pad
        end
        
        def forward(x)
          # Apply padding if needed
          if @padding[0] > 0
            # Add padding to the last dimension (width)
            padding_array = [[0, 0], [0, 0], [@padding[0], @padding[0]]]
            x = MLX.pad(x, padding_array)
          end
          
          # Apply average pooling
          MLX.avg_pool1d(x, @kernel_size[0], stride: @stride[0], 
                        ceil_mode: @ceil_mode, 
                        count_include_pad: @count_include_pad)
        end
      end

      # Average pooling operation in 2D
      class AvgPool2d < BasePoolNd
        attr_reader :count_include_pad
        
        def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false, count_include_pad: true)
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = stride.nil? ? nil : [stride].flatten
          padding = [padding].flatten
          
          # Ensure all parameter arrays have length 2
          kernel_size = kernel_size.length == 1 ? [kernel_size[0], kernel_size[0]] : kernel_size[0..1]
          stride = if stride.nil?
                    kernel_size.dup
                  else
                    stride.length == 1 ? [stride[0], stride[0]] : stride[0..1]
                  end
          padding = padding.length == 1 ? [padding[0], padding[0]] : padding[0..1]
          
          super(kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
          @count_include_pad = count_include_pad
        end
        
        def forward(x)
          # Apply padding if needed
          if @padding[0] > 0 || @padding[1] > 0
            # Add padding to the last two dimensions (height, width)
            padding_array = [[0, 0], [0, 0], [@padding[0], @padding[0]], [@padding[1], @padding[1]]]
            x = MLX.pad(x, padding_array)
          end
          
          # Apply average pooling
          MLX.avg_pool2d(x, @kernel_size, stride: @stride, 
                        ceil_mode: @ceil_mode,
                        count_include_pad: @count_include_pad)
        end
      end

      # Average pooling operation in 3D
      class AvgPool3d < BasePoolNd
        attr_reader :count_include_pad
        
        def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false, count_include_pad: true)
          # Convert scalar parameters to arrays
          kernel_size = [kernel_size].flatten
          stride = stride.nil? ? nil : [stride].flatten
          padding = [padding].flatten
          
          # Ensure all parameter arrays have length 3
          kernel_size = case kernel_size.length
                        when 1 then [kernel_size[0], kernel_size[0], kernel_size[0]]
                        when 2 then [kernel_size[0], kernel_size[1], kernel_size[1]]
                        else kernel_size[0..2]
                        end
                        
          stride = if stride.nil?
                    kernel_size.dup
                  else
                    case stride.length
                    when 1 then [stride[0], stride[0], stride[0]]
                    when 2 then [stride[0], stride[1], stride[1]]
                    else stride[0..2]
                    end
                  end
                  
          padding = case padding.length
                    when 1 then [padding[0], padding[0], padding[0]]
                    when 2 then [padding[0], padding[1], padding[1]]
                    else padding[0..2]
                    end
          
          super(kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
          @count_include_pad = count_include_pad
        end
        
        def forward(x)
          # Apply padding if needed
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
          
          # Apply average pooling
          MLX.avg_pool3d(x, @kernel_size, stride: @stride, 
                        ceil_mode: @ceil_mode,
                        count_include_pad: @count_include_pad)
        end
      end

      # Global average pooling - reduces spatial dimensions to 1x1
      class AdaptiveAvgPool1d < MLX::NN::Module
        attr_reader :output_size
        
        def initialize(output_size)
          super()
          @output_size = output_size
        end
        
        def forward(x)
          # Get input dimensions
          batch_size, channels, width = x.shape
          
          # Calculate stride and kernel size
          if @output_size == 1
            # Special case: global average pooling
            kernel_size = width
            stride = 1
          else
            stride = (width / @output_size.to_f).floor
            kernel_size = width - (@output_size - 1) * stride
          end
          
          # Apply average pooling
          MLX.avg_pool1d(x, kernel_size, stride: stride)
        end
      end

      # Adaptive average pooling for 2D inputs
      class AdaptiveAvgPool2d < MLX::NN::Module
        attr_reader :output_size
        
        def initialize(output_size)
          super()
          @output_size = output_size.is_a?(Array) ? output_size : [output_size, output_size]
        end
        
        def forward(x)
          # Get input dimensions
          batch_size, channels, height, width = x.shape
          
          # Calculate parameters for both dimensions
          out_h, out_w = @output_size
          
          if out_h == 1 && out_w == 1
            # Global average pooling
            return MLX.mean(x, axis: [2, 3], keepdim: true)
          end
          
          # Use regions of approximately equal size
          stride_h = (height / out_h.to_f).floor
          kernel_h = height - (out_h - 1) * stride_h
          
          stride_w = (width / out_w.to_f).floor
          kernel_w = width - (out_w - 1) * stride_w
          
          # Apply average pooling
          MLX.avg_pool2d(x, [kernel_h, kernel_w], stride: [stride_h, stride_w])
        end
      end

      # Adaptive average pooling for 3D inputs
      class AdaptiveAvgPool3d < MLX::NN::Module
        attr_reader :output_size
        
        def initialize(output_size)
          super()
          @output_size = if output_size.is_a?(Array)
                          output_size
                        else
                          [output_size, output_size, output_size]
                        end
        end
        
        def forward(x)
          # Get input dimensions
          batch_size, channels, depth, height, width = x.shape
          
          # Calculate parameters for all three dimensions
          out_d, out_h, out_w = @output_size
          
          if out_d == 1 && out_h == 1 && out_w == 1
            # Global average pooling
            return MLX.mean(x, axis: [2, 3, 4], keepdim: true)
          end
          
          # Use regions of approximately equal size
          stride_d = (depth / out_d.to_f).floor
          kernel_d = depth - (out_d - 1) * stride_d
          
          stride_h = (height / out_h.to_f).floor
          kernel_h = height - (out_h - 1) * stride_h
          
          stride_w = (width / out_w.to_f).floor
          kernel_w = width - (out_w - 1) * stride_w
          
          # Apply average pooling
          MLX.avg_pool3d(x, [kernel_d, kernel_h, kernel_w], 
                         stride: [stride_d, stride_h, stride_w])
        end
      end
    end
  end
end 