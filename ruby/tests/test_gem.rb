#!/usr/bin/env ruby

require_relative 'mlx_test_case'

# Define a class for running gem tests
class TestGem < MLXTestCase
  def test_gem_loading
    puts "\nMLX Ruby Bindings Test"
    puts "=====================\n"
    
    puts "MLX Core loaded: #{defined?(MLX::Core) ? 'Yes' : 'No'}"
    puts "MLX Version: #{MLX::VERSION}\n"

    # Test that all modules in MLX::Core have constants in MLX
    puts "Testing Module Structure"
    puts "---------------------"
    
    modules_count = 0
    modules_success = 0
    
    MLX::Core.modules.each do |mod|
      modules_count += 1
      module_name = mod.name.split('::').last
      full_const_path = "MLX::#{module_name}"
      
      mlx_has_const = Object.const_defined?(full_const_path)
      if mlx_has_const
        mlx_const = Object.const_get(full_const_path)
        const_matches = (mlx_const == mod)
      else
        const_matches = false
      end
      
      # Get module methods that can be exposed statically
      methods = mod.methods(false).reject { |m| [:inherited, :included, :extended].include?(m) }
      
      # Count methods also available on MLX
      mlx_methods = methods.select { |method| MLX.respond_to?(method) }
      methods_exposed = mlx_methods.size
      
      result = mlx_has_const && const_matches && methods_exposed > 0
      modules_success += 1 if result
      
      puts "#{mod.name}: #{result ? 'Success' : 'Failure'}"
      puts "  - MLX has constant: #{mlx_has_const ? 'Yes' : 'No'}"
      puts "  - Constant matches: #{const_matches ? 'Yes' : 'No'}"
      puts "  - #{methods_exposed}/#{methods.size} methods exposed on MLX"
    end
    
    # Test that all classes in MLX::Core have constants in MLX
    puts "\nTesting Class Structure"
    puts "---------------------"
    
    classes_count = 0
    classes_success = 0
    
    MLX::Core.classes.each do |klass|
      classes_count += 1
      class_name = klass.name.split('::').last
      full_const_path = "MLX::#{class_name}"
      
      # For Array, check for MlxArray instead
      if class_name == "Array"
        full_const_path = "MLX::MlxArray"
      end
      
      # Check if corresponding constant exists in MLX
      mlx_has_const = Object.const_defined?(full_const_path)
      if mlx_has_const
        mlx_const = Object.const_get(full_const_path)
        const_matches = (mlx_const == klass)
        
        if !const_matches
          puts "#{klass.name}: Failure"
          puts "  - MLX has constant: Yes"
          puts "  - Constant matches: No"
          puts "  - MLX::Core class: #{klass.inspect} (#{klass.object_id})"
          puts "  - MLX class: #{mlx_const.inspect} (#{mlx_const.object_id})"
        else
          # Get class methods that can be exposed statically
          methods = klass.methods(false).reject { |m| [:inherited, :included, :extended].include?(m) }
          
          # Count methods also available on MLX
          mlx_methods = methods.select { |method| MLX.respond_to?(method) }
          methods_exposed = mlx_methods.size
          
          result = mlx_has_const && const_matches && (methods.empty? || methods_exposed > 0)
          classes_success += 1 if result
          
          puts "#{klass.name}: #{result ? 'Success' : 'Failure'}"
          puts "  - MLX has constant: Yes"
          puts "  - Constant matches: Yes"
          puts "  - #{methods_exposed}/#{methods.size} methods exposed on MLX"
        end
      else
        const_matches = false
        puts "#{klass.name}: Failure"
        puts "  - MLX has constant: No"
        puts "  - Constant matches: No"
      end
    end
    
    # Summary
    puts "\nSummary"
    puts "-------"
    puts "Modules: #{modules_success}/#{modules_count} correctly exposed"
    puts "Classes: #{classes_success}/#{classes_count} correctly exposed"
    
    # Test a few specific methods to ensure they work
    puts "\nTesting Sample Methods"
    puts "-------------------"
    
    methods_to_test = [
      [:zeros, [2, 2], MLX::Core::FLOAT32],
      [:ones, [3, 3], MLX::Core::FLOAT32],
      [:array, [[1, 2], [3, 4]]]
    ]
    
    methods_to_test.each do |method, *args|
      begin
        result = MLX.send(method, *args)
        puts "MLX.#{method}: Success - returned #{result.class.name}"
      rescue => e
        puts "MLX.#{method}: Failure - #{e.message}"
      end
    end
  end

  private
  
end

# If running this file directly, run the test
if __FILE__ == $0
  test = TestGem.new(:test_gem_loading)
  test.run
end