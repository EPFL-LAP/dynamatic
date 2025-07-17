vlib work
vmap work work
project new . simulation work modelsim.ini 0
project open simulation
project addfile ../VHDL_SRC/sink.vhd
project addfile ../VHDL_SRC/sink_dataless.vhd
project addfile ../VHDL_SRC/handshake_constant_3.vhd
project addfile ../VHDL_SRC/muli.vhd
project addfile ../VHDL_SRC/handshake_cmpi_0.vhd
project addfile ../VHDL_SRC/tehb_dataless.vhd
project addfile ../VHDL_SRC/control_merge_dataless.vhd
project addfile ../VHDL_SRC/handshake_constant_0.vhd
project addfile ../VHDL_SRC/single_argument.vhd
project addfile ../VHDL_SRC/eager_fork_register_block.vhd
project addfile ../VHDL_SRC/handshake_fork.vhd
project addfile ../VHDL_SRC/tehb.vhd
project addfile ../VHDL_SRC/two_port_RAM.vhd
project addfile ../VHDL_SRC/hls_verify_iterative_sqrt_tb.vhd
project addfile ../VHDL_SRC/extsi.vhd
project addfile ../VHDL_SRC/oehb.vhd
project addfile ../VHDL_SRC/handshake_cmpi_2.vhd
project addfile ../VHDL_SRC/delay_buffer.vhd
project addfile ../VHDL_SRC/source.vhd
project addfile ../VHDL_SRC/merge_dataless.vhd
project addfile ../VHDL_SRC/cond_br_dataless.vhd
project addfile ../VHDL_SRC/shrsi.vhd
project addfile ../VHDL_SRC/oehb_dataless.vhd
project addfile ../VHDL_SRC/iterative_sqrt.vhd
project addfile ../VHDL_SRC/handshake_cmpi_3.vhd
project addfile ../VHDL_SRC/handshake_constant_1.vhd
project addfile ../VHDL_SRC/addi.vhd
project addfile ../VHDL_SRC/iterative_sqrt_wrapper.vhd
project addfile ../VHDL_SRC/logic.vhd
project addfile ../VHDL_SRC/andi.vhd
project addfile ../VHDL_SRC/simpackage.vhd
project addfile ../VHDL_SRC/handshake_cmpi_4.vhd
project addfile ../VHDL_SRC/cond_br.vhd
project addfile ../VHDL_SRC/join.vhd
project addfile ../VHDL_SRC/types.vhd
project addfile ../VHDL_SRC/tb_join.vhd
project addfile ../VHDL_SRC/mux.vhd
project addfile ../VHDL_SRC/ori.vhd
project addfile ../VHDL_SRC/fork_dataless.vhd
project addfile ../VHDL_SRC/handshake_constant_2.vhd
project addfile ../VHDL_SRC/handshake_cmpi_1.vhd
project calculateorder
project compileall
eval vsim iterative_sqrt_wrapper_tb
log -r *
run -all
exit
