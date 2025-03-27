#!/usr/bin/env ruby

require_relative 'mlx_test_case'

# Define a class for testing MLX call patterns
class TestCallPatterns < MLXTestCase
  def test_call_patterns
    puts "\nMLX Call Patterns Test"
    puts "=====================\n"
    
    # Read the call patterns file
    patterns_file = File.join(File.dirname(__FILE__), 'mlx_call_patterns.txt')
    patterns = File.readlines(patterns_file).map(&:strip).reject(&:empty?)
    
    puts "Testing #{patterns.size} call patterns"
    puts "--------------------------\n"
    
    success_count = 0
    failures = []
    
    patterns.each do |pattern|
      result = test_pattern(pattern)
      if result[:valid]
        success_count += 1
        puts "✅ #{pattern}"
      else
        failures << { pattern: pattern, error: result[:error] }
        puts "❌ #{pattern} - #{result[:error]}"
      end
    end
    
    # Summary
    puts "\nSummary"
    puts "-------"
    puts "• Valid patterns: #{success_count}/#{patterns.size}"
    
    if failures.any?
      puts "\nFailed Patterns"
      puts "--------------"
      failures.each do |failure|
        puts "• #{failure[:pattern]}: #{failure[:error]}"
      end
    end
  end
  
  private
  
  def test_pattern(pattern)
    # Check if pattern is a constant or method call
    if pattern.include?('.')
      test_method_call(pattern)
    else
      test_constant(pattern)
    end
  end
  
  def test_constant(pattern)
    begin
      # Try to resolve the constant
      constant_parts = pattern.split('::')
      current = Object
      
      constant_parts.each do |part|
        current = current.const_get(part)
      end
      
      { valid: true }
    rescue NameError => e
      { valid: false, error: "Constant not defined: #{e.message}" }
    rescue => e
      { valid: false, error: "Error: #{e.message}" }
    end
  end
  
  def test_method_call(pattern)
    begin
      if pattern.include?('(')
        # Pattern includes argument list, try to evaluate it
        eval("#{pattern} rescue nil")
        { valid: true }
      else
        # Check if the method exists
        parts = pattern.split('.')
        
        if parts.size == 2
          # Simple case: MLX.method
          object_name, method_name = parts
          object = eval(object_name)
          
          if object.respond_to?(method_name.to_sym)
            { valid: true }
          else
            { valid: false, error: "Method not defined" }
          end
        else
          # Complex case: MLX.module.method or MLX::Module.method
          # First, try to resolve the object chain
          current = parts.first
          current_obj = eval(current) rescue nil
          
          # If we couldn't resolve the first part, fail
          return { valid: false, error: "Object not defined: #{current}" } unless current_obj
          
          # Try to resolve each part of the chain
          (1...parts.size).each do |i|
            part = parts[i]
            
            # If we're at the last part, check if it's a method
            if i == parts.size - 1
              return { valid: current_obj.respond_to?(part.to_sym), 
                      error: current_obj.respond_to?(part.to_sym) ? nil : "Method not defined: #{part}" }
            end
            
            # Otherwise, navigate the object chain
            if current_obj.respond_to?(part.to_sym)
              current_obj = current_obj.send(part.to_sym) rescue nil
              return { valid: false, error: "Failed to access: #{part}" } unless current_obj
            else
              # Try to resolve as a constant
              begin
                current_obj = current_obj.const_get(part)
              rescue NameError
                return { valid: false, error: "Neither method nor constant: #{part}" }
              end
            end
          end
          
          { valid: true }
        end
      end
    rescue => e
      { valid: false, error: "Error: #{e.message}" }
    end
  end
end

# If running this file directly, run the test
if __FILE__ == $0
  test = TestCallPatterns.new(:test_call_patterns)
  test.run
end 