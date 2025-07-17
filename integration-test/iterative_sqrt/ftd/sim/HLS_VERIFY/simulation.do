vlib work
vmap work work
project new . simulation work modelsim.ini 0
project open simulation
project addfile ../VHDL_SRC/single_argument.vhd
project addfile ../VHDL_SRC/two_port_RAM.vhd
project addfile ../VHDL_SRC/hls_verify_iterative_sqrt_tb.vhd
project addfile ../VHDL_SRC/simpackage.vhd
project addfile ../VHDL_SRC/tb_join.vhd
project calculateorder
project compileall
eval vsim iterative_sqrt_wrapper_tb
log -r *
run -all
exit
