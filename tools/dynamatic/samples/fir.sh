# Sample sequence of commands for Dynamatic frontend

# Indicate the path to Dynamatic's top-level directory here (leave unchanged if
# running the frontend from the top-level directory)
set-dynamatic-path  .

# Indicate the path the legacy Dynamatic's top-level directory here (required
# for write-hdl and simulate)
set-legacy-path     ../dynamatic-utils/legacy-dynamatic/dhls/etc/dynamatic

# Set the source file to run (kernel must have the same name as the filename,
# without the extension)
set-src             integration-test/fir/fir.c

# Compile (from source to Handshake IR/DOT)
# Remove the flag to run smart buffer placement (requires Gurobi)
compile             --simple-buffers

# Generate the VHDL for the dataflow circuit
write-hdl

# Simulate using Modelsim
simulate

# Synthesize using Vivado
synthesize

# Exit the frontend
exit
