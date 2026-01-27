################################################################
# BASE SYNTHESIS TEMPLATE
################################################################
base_synthesis_tcl = r"""
# Date: %date
# Define global variable
set TOP_DESIGN %design
set VHDL_SRC %hdlsrc

# Read all source files
%inputs

# Read the design constraints
read_xdc period.xdc

# Run synthesis
synth_design -top $TOP_DESIGN -part xc7k160tfbg484-1 -no_iobuf -mode out_of_context -flatten_hierarchy none

# Write out the vhdl file
write_vhdl -force ./${TOP_DESIGN}_syn.vhd

# Ciao!
exit
"""
