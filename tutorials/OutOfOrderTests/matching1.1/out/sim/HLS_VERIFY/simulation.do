vlib work
vmap work work
project new . simulation work modelsim.ini 0
project open simulation
project addfile ../VHDL_SRC/matching.vhd
project addfile ../VHDL_SRC/single_argument.vhd
project addfile ../VHDL_SRC/two_port_RAM.vhd
project addfile ../VHDL_SRC/hls_verify_matching_tb.vhd
project addfile ../VHDL_SRC/matching_wrapper.vhd
project addfile ../VHDL_SRC/simpackage.vhd
project addfile ../VHDL_SRC/tb_join.vhd
project calculateorder
project compileall
eval vsim matching_wrapper_tb
log -r *
run 10000ns -all
exit
