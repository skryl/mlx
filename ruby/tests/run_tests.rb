#!/usr/bin/env ruby

require 'minitest/autorun'

# Load all test files
Dir.glob(File.join(File.dirname(__FILE__), 'test_*.rb')).each do |file|
  require file
end

# This script will automatically run all tests when executed.
# Run with:
#   ruby run_tests.rb
#
# For verbose output:
#   ruby run_tests.rb -v
#
# To run a specific test file:
#   ruby -I. test_array.rb
#
# To run a specific test:
#   ruby -I. test_array.rb -n test_array_creation 