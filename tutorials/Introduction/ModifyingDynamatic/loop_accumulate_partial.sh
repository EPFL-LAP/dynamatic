# Set the source file to run
set-src             tutorials/Introduction/UsingDynamatic/loop_accumulate.c

# Generate the VHDL design corresponding to the source code
write-hdl

# Simulate and verify using Modelsim
simulate

# Prepare data for the dataflow visualizer
visualize

# Exit the frontend
exit
