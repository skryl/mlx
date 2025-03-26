require 'json'
require 'ipaddr'
require 'optparse'
require 'tempfile'
require 'open3'
require 'fileutils'
require 'socket'
require 'securerandom'

module MLX
  # Module for distributed job launching and configuration
  module DistributedRun
    # Class representing a host in the distributed environment
    class Host
      attr_reader :rank, :ssh_hostname, :ips
      
      def initialize(rank, ssh_hostname, ips = [])
        @rank = rank
        @ssh_hostname = ssh_hostname
        @ips = ips
      end
    end
    
    # Class representing a Thunderbolt port
    class ThunderboltPort
      attr_reader :iface, :uuid, :connected_to
      
      def initialize(iface, uuid, connected_to = nil)
        @iface = iface
        @uuid = uuid
        @connected_to = connected_to
      end
    end
    
    # Class representing a Thunderbolt host
    class ThunderboltHost
      attr_reader :name, :ports
      
      def initialize(name, ports = [])
        @name = name
        @ports = ports
      end
    end
    
    # Parse hardware ports information
    # 
    # @param ports_string [String] String containing ports information
    # @return [Hash] Mapping of port names to devices
    def self.parse_hardware_ports(ports_string)
      ports = {}
      port_name = nil
      
      ports_string.encode('UTF-8', invalid: :replace).split("\n").each do |line|
        if line.start_with?("Hardware Port:")
          port_name = line.strip[15..-1]
        elsif line.start_with?("Device:") && port_name
          ports[port_name] = line.strip[8..-1]
          port_name = nil
        end
      end
      
      ports
    end
    
    # Extract communication rings from a set of hosts
    # 
    # @param hosts [Array<ThunderboltHost>] List of hosts
    # @param index [Hash] Mapping of UUIDs to (host_index, port_index) pairs
    # @return [Array<Array>] List of communication rings
    def self.extract_rings(hosts, index)
      # Check if a port can be used
      usable_port = lambda do |i, j, used_ports|
        !used_ports.include?([i, j]) && hosts[i].ports[j].connected_to
      end
      
      # Depth-first search to find cycles
      dfs = lambda do |start_node, node, path, visited, used_ports|
        path << node
        visited << node
        
        hosts[node].ports.each_with_index do |port, j|
          next unless usable_port.call(node, j, used_ports)
          
          next_node, _ = index[port.connected_to]
          if next_node == start_node
            yield path.dup
          elsif !visited.include?(next_node)
            dfs.call(start_node, next_node, path, visited, used_ports) { |result| yield result }
          end
        end
        
        path.pop
        visited.delete(node)
      end
      
      # Concretize a cycle into actual port pairs
      concretize = lambda do |cycle, used_ports|
        concrete_path = []
        
        cycle.zip(cycle[1..-1] + cycle[0..0]).each do |n1, n2|
          found = false
          
          hosts[n1].ports.each_with_index do |port, j|
            next unless usable_port.call(n1, j, used_ports)
            
            n2_hat, nj = index[port.connected_to]
            if n2 == n2_hat
              concrete_path << [([n1, j]), ([n2, nj])]
              used_ports << [n1, j]
              used_ports << [n2, nj]
              found = true
              break
            end
          end
          
          raise RuntimeError, "Couldn't concretize the cycle" unless found
        end
        
        concrete_path
      end
      
      # Normalize cycles to have a consistent direction
      normalize = lambda do |path|
        small_to_large = path.count { |p| p[0][0] < p[1][0] }
        
        if small_to_large > path.length - small_to_large
          path
        else
          path.map { |p| [p[1], p[0]] }
        end
      end
      
      # Extract all possible rings
      rings = []
      used_ports = Set.new
      
      hosts.length.times do |start_node|
        loop do
          ring = []
          
          dfs.call(start_node, start_node, [], Set.new, used_ports) do |r|
            ring = r if r.length > ring.length
            break if ring.length == hosts.length # Won't find a bigger ring
          end
          
          break if ring.empty?
          
          begin
            rings << normalize.call(concretize.call(ring, used_ports))
          rescue RuntimeError
            return rings if rings.any?
            raise
          end
        end
      end
      
      rings
    end
    
    # Log a message if verbose mode is enabled
    # 
    # @param verbose [Boolean] Whether to output the message
    # @param args [Array] Message parts
    # @param kwargs [Hash] Additional options for puts
    def self.log(verbose, *args, **kwargs)
      return unless verbose
      puts "\033[32m[INFO] #{args.join(' ')}\033[0m", **kwargs
    end
    
    # Log a warning message
    # 
    # @param args [Array] Message parts
    # @param kwargs [Hash] Additional options for puts
    def self.log_warning(*args, **kwargs)
      kwargs[:out] = STDERR
      puts "\033[33m[WARN] #{args.join(' ')}\033[0m", **kwargs
    end
    
    # Log an error message
    # 
    # @param args [Array] Message parts
    # @param kwargs [Hash] Additional options for puts
    def self.log_error(*args, **kwargs)
      kwargs[:out] = STDERR
      puts "\033[31m[ERROR] #{args.join(' ')}\033[0m", **kwargs
    end
    
    # Parse a hostfile containing host information
    # 
    # @param hostfile [String] Path to the hostfile
    # @return [Array<Host>] List of hosts
    def self.parse_hostfile(hostfile)
      hostfile_path = File.expand_path(hostfile)
      
      unless File.exist?(hostfile_path)
        raise ArgumentError, "Hostfile #{hostfile_path} doesn't exist"
      end
      
      begin
        hosts = []
        JSON.parse(File.read(hostfile_path)).each_with_index do |h, i|
          hosts << Host.new(i, h["ssh"], h["ips"] || [])
        end
        hosts
      rescue => e
        raise ArgumentError, "Failed to parse hostfile #{hostfile_path}: #{e.message}"
      end
    end
    
    # Parse a comma-separated list of hosts
    # 
    # @param hostlist [String] Comma-separated list of hosts
    # @param repeats [Integer] Number of processes per host
    # @return [Array<Host>] List of hosts
    def self.parse_hostlist(hostlist, repeats)
      hosts = []
      rank = 0
      
      hostlist.split(",").each do |h|
        if h.empty?
          raise ArgumentError, "Hostname cannot be empty"
        end
        
        # Check if it's an IP address
        begin
          IPAddr.new(h)
          ips = [h]
        rescue IPAddr::InvalidAddressError
          ips = []
        end
        
        repeats.times do
          hosts << Host.new(rank, h, ips)
          rank += 1
        end
      end
      
      hosts
    end
    
    # Create a monitoring script for a distributed job
    # 
    # @param rank [Integer] Process rank
    # @param hostfile [String] Path to the hostfile
    # @param cwd [String, nil] Working directory
    # @param env [Array<String>] Environment variables
    # @param command [Array<String>] Command to run
    # @param verbose [Boolean] Whether to enable verbose output
    # @return [String] Python script content
    def self.make_monitor_script(rank, hostfile, cwd, env, command, verbose)
      script = ""
      
      # Imports
      script += "import os\n"
      script += "import sys\n"
      script += "import tempfile\n"
      script += "from pathlib import Path\n"
      
      # Write PID to file
      script += "_, pidfile = tempfile.mkstemp()\n"
      script += "open(pidfile, 'w').write(str(os.getpid()))\n"
      script += "print(pidfile, flush=True)\n"
      
      # Change working directory
      d = cwd || Dir.pwd
      script += "if Path(#{d.inspect}).exists():\n"
      script += "    os.chdir(#{d.inspect})\n"
      
      if cwd
        script += "else:\n"
        script += "    print('Failed to change directory to', #{d.inspect}, file=sys.stderr)\n"
        script += "    sys.exit(1)\n"
      end
      
      # Set environment variables
      script += "env = dict(os.environ)\n"
      
      env.each do |e|
        key, *value = e.split("=", 2)
        value_str = value.first || ""
        
        # Skip invalid environment variables
        unless key =~ /^[a-zA-Z0-9_]+$/
          log_warning("'#{e}' is an invalid environment variable so it is ignored")
          next
        end
        
        script += "env[#{key.inspect}] = #{value_str.inspect}\n"
      end
      
      # Setup distributed environment variables
      if !hostfile.empty?
        script += "_, hostfile = tempfile.mkstemp()\n"
        script += "with open(hostfile, 'w') as f:\n"
        script += "    f.write(#{hostfile.inspect})\n"
        script += "env['MLX_RING_VERBOSE'] = '1'\n" if verbose
        script += "env['MLX_HOSTFILE'] = hostfile\n"
        script += "env['MLX_RANK'] = '#{rank}'\n"
      end
      
      # Execute the command
      script += "command = [#{command.map(&:inspect).join(',')}]\n"
      script += "os.execve(command[0], command, env)\n"
      
      script
    end
    
    # Launch a distributed job using the ring backend
    # 
    # @param hosts [Array<Host>] List of hosts
    # @param args [Hash] Command-line arguments
    # @param command [Array<String>] Command to run
    # @return [Boolean] Whether the job completed successfully
    def self.launch_ring(hosts, args, command)
      stop = false
      exit_codes = [nil] * hosts.length
      
      # Setup signal handling
      trap("INT") { stop = true }
      trap("TERM") { stop = true }
      
      # Convert hostfile to JSON
      hostfile_content = JSON.generate(
        hosts.map do |h|
          { "rank" => h.rank, "ips" => h.ips }
        end
      )
      
      # Print host information
      if args[:verbose]
        hosts.each do |h|
          puts "#{h.rank}: #{h.ssh_hostname}"
        end
      end
      
      # Launch processes
      processes = []
      
      hosts.each do |host|
        # Create Python script
        monitor_script = make_monitor_script(
          host.rank,
          hostfile_content,
          args[:cwd],
          args[:env] || [],
          command,
          args[:verbose]
        )
        
        # Create temp file for the script
        script_file = Tempfile.new(["mlx_monitor", ".py"])
        script_file.write(monitor_script)
        script_file.close
        
        # Build the SSH command
        if host.ssh_hostname == "localhost"
          cmd = ["python3", script_file.path]
        else
          cmd = ["ssh", "-o", "StrictHostKeyChecking=no", host.ssh_hostname, "python3 -"]
        end
        
        # Start the process
        log(args[:verbose], "Launching rank #{host.rank} on #{host.ssh_hostname}")
        
        if host.ssh_hostname == "localhost"
          process = IO.popen(cmd, "r")
          processes << { rank: host.rank, process: process, pidfile: nil }
        else
          process = IO.popen(cmd, "w+")
          process.write(monitor_script)
          process.close_write
          processes << { rank: host.rank, process: process, pidfile: nil }
        end
      end
      
      # Get PID files
      begin
        Timeout.timeout(5) do
          processes.each do |p|
            p[:pidfile] = p[:process].gets.strip
          end
        end
      rescue Timeout::Error
        log_error("Timeout waiting for processes to start")
        cleanup_processes(processes)
        return false
      end
      
      # Wait for processes to complete
      while processes.any? && !stop
        # Check if any process has output
        ready_processes = processes.select { |p| IO.select([p[:process]], nil, nil, 0.1) }
        
        ready_processes.each do |p|
          begin
            line = p[:process].gets
            if line
              puts "#{p[:rank]}: #{line}"
            else
              # Process ended, get exit code
              _, status = Process.waitpid2(p[:process].pid)
              exit_codes[p[:rank]] = status.exitstatus
              processes.delete(p)
            end
          rescue IOError
            # Process closed
            processes.delete(p)
          end
        end
        
        # Sleep briefly to avoid busy-waiting
        sleep(0.01)
      end
      
      # Cleanup if stopped early
      if stop || processes.any?
        log(args[:verbose], "Cleaning up processes")
        cleanup_processes(processes)
      end
      
      # Return success if all processes exited with code 0
      exit_codes.all? { |code| code == 0 }
    end
    
    # Cleanup processes
    # 
    # @param processes [Array<Hash>] List of processes
    def self.cleanup_processes(processes)
      processes.each do |p|
        # Kill process if it's still running
        begin
          Process.kill("TERM", p[:process].pid)
        rescue Errno::ESRCH
          # Process already terminated
        end
        
        # Remove PID file
        if p[:pidfile] && File.exist?(p[:pidfile])
          File.unlink(p[:pidfile])
        end
      end
    end
    
    # Launch a distributed job
    # 
    # @param args [Hash] Command-line arguments
    # @param command [Array<String>] Command to run
    # @return [Boolean] Whether the job completed successfully
    def self.launch(args, command)
      # Parse hosts
      hosts = if args[:hostfile]
        parse_hostfile(args[:hostfile])
      elsif args[:hostlist]
        parse_hostlist(args[:hostlist], args[:npernode] || 1)
      else
        [Host.new(0, "localhost")]
      end
      
      # Launch using the appropriate backend
      case args[:backend]
      when "ring"
        launch_ring(hosts, args, command)
      when "mpi"
        log_error("MPI backend not yet implemented")
        false
      else
        log_error("Unknown backend: #{args[:backend]}")
        false
      end
    end
    
    # Configure hosts for distributed training
    # 
    # @param args [Hash] Command-line arguments
    # @return [Boolean] Whether configuration was successful
    def self.configure(args)
      # Parse hosts
      hosts = if args[:hostfile]
        parse_hostfile(args[:hostfile])
      elsif args[:hostlist]
        parse_hostlist(args[:hostlist], 1)
      else
        [Host.new(0, "localhost")]
      end
      
      # Configure based on backend
      case args[:backend]
      when "ring"
        configure_ring(hosts, args)
      when "mpi"
        log_error("MPI backend configuration not yet implemented")
        false
      else
        log_error("Unknown backend: #{args[:backend]}")
        false
      end
    end
    
    # Configure hosts for ring communication
    # 
    # @param hosts [Array<Host>] List of hosts
    # @param args [Hash] Command-line arguments
    # @return [Boolean] Whether configuration was successful
    def self.configure_ring(hosts, args)
      log(args[:verbose], "Configuring ring communication for #{hosts.length} hosts")
      
      # TODO: Implement ring configuration
      # This would involve:
      # 1. Detecting Thunderbolt connections
      # 2. Establishing the optimal ring topology
      # 3. Configuring each host
      
      # For now, just print a message
      puts "Ring configuration would be performed here for hosts: #{hosts.map(&:ssh_hostname).join(', ')}"
      true
    end
    
    # Main entry point for the mlx.launch command
    def self.main
      options = {}
      
      parser = OptionParser.new do |opts|
        opts.banner = "Usage: mlx.launch [options] command..."
        
        opts.on("-h", "--hostfile FILE", "File with hosts to use") do |file|
          options[:hostfile] = file
        end
        
        opts.on("-H", "--hosts LIST", "Comma-separated list of hosts") do |list|
          options[:hostlist] = list
        end
        
        opts.on("-n", "--npernode N", Integer, "Number of processes per node") do |n|
          options[:npernode] = n
        end
        
        opts.on("-b", "--backend BACKEND", "Distributed backend to use (ring, mpi)") do |backend|
          options[:backend] = backend
        end
        
        opts.on("-e", "--env VAR=VALUE", "Environment variable to set") do |env|
          options[:env] ||= []
          options[:env] << env
        end
        
        opts.on("-d", "--cwd DIR", "Working directory") do |dir|
          options[:cwd] = dir
        end
        
        opts.on("-v", "--verbose", "Verbose output") do
          options[:verbose] = true
        end
        
        opts.on("--help", "Show this message") do
          puts opts
          exit
        end
      end
      
      # Parse options
      begin
        parser.parse!
      rescue OptionParser::InvalidOption => e
        log_error(e.message)
        puts parser
        exit(1)
      end
      
      # Get command
      command = ARGV
      
      if command.empty?
        log_error("No command specified")
        puts parser
        exit(1)
      end
      
      # Set defaults
      options[:backend] ||= "ring"
      
      # Launch the command
      success = launch(options, command)
      
      # Exit with appropriate code
      exit(success ? 0 : 1)
    end
    
    # Main entry point for the mlx.distributed_config command
    def self.distributed_config
      options = {}
      
      parser = OptionParser.new do |opts|
        opts.banner = "Usage: mlx.distributed_config [options]"
        
        opts.on("-h", "--hostfile FILE", "File with hosts to configure") do |file|
          options[:hostfile] = file
        end
        
        opts.on("-H", "--hosts LIST", "Comma-separated list of hosts") do |list|
          options[:hostlist] = list
        end
        
        opts.on("-b", "--backend BACKEND", "Distributed backend to configure (ring, mpi)") do |backend|
          options[:backend] = backend
        end
        
        opts.on("-v", "--verbose", "Verbose output") do
          options[:verbose] = true
        end
        
        opts.on("--help", "Show this message") do
          puts opts
          exit
        end
      end
      
      # Parse options
      begin
        parser.parse!
      rescue OptionParser::InvalidOption => e
        log_error(e.message)
        puts parser
        exit(1)
      end
      
      # Set defaults
      options[:backend] ||= "ring"
      
      # Configure the hosts
      success = configure(options)
      
      # Exit with appropriate code
      exit(success ? 0 : 1)
    end
  end
end 