vlib work
vmap work work
project new . simulation work modelsim.ini 0
project open simulation
project addfile ../VHDL_SRC/sink.vhd
project addfile ../VHDL_SRC/load.vhd
project addfile ../VHDL_SRC/multiple_exit.vhd
project addfile ../VHDL_SRC/handshake_constant_5.vhd
project addfile ../VHDL_SRC/handshake_constant_3.vhd
project addfile ../VHDL_SRC/handshake_cmpi_0.vhd
project addfile ../VHDL_SRC/tehb_dataless.vhd
project addfile ../VHDL_SRC/control_merge_dataless.vhd
project addfile ../VHDL_SRC/selector.vhd
project addfile ../VHDL_SRC/store.vhd
project addfile ../VHDL_SRC/trunci.vhd
project addfile ../VHDL_SRC/handshake_constant_0.vhd
project addfile ../VHDL_SRC/mem_to_bram.vhd
project addfile ../VHDL_SRC/single_argument.vhd
project addfile ../VHDL_SRC/handshake_lsq_lsq1.vhd
project addfile ../VHDL_SRC/eager_fork_register_block.vhd
project addfile ../VHDL_SRC/handshake_fork.vhd
project addfile ../VHDL_SRC/tehb.vhd
project addfile ../VHDL_SRC/two_port_RAM.vhd
project addfile ../VHDL_SRC/extsi.vhd
project addfile ../VHDL_SRC/oehb.vhd
project addfile ../VHDL_SRC/handshake_cmpi_2.vhd
project addfile ../VHDL_SRC/source.vhd
project addfile ../VHDL_SRC/merge_dataless.vhd
project addfile ../VHDL_SRC/handshake_constant_4.vhd
project addfile ../VHDL_SRC/handshake_lsq_lsq1_core.vhd
project addfile ../VHDL_SRC/cond_br_dataless.vhd
project addfile ../VHDL_SRC/oehb_dataless.vhd
project addfile ../VHDL_SRC/lazy_fork_dataless.vhd
project addfile ../VHDL_SRC/handshake_constant_1.vhd
project addfile ../VHDL_SRC/addi.vhd
project addfile ../VHDL_SRC/logic.vhd
project addfile ../VHDL_SRC/andi.vhd
project addfile ../VHDL_SRC/simpackage.vhd
project addfile ../VHDL_SRC/cond_br.vhd
project addfile ../VHDL_SRC/join.vhd
project addfile ../VHDL_SRC/types.vhd
project addfile ../VHDL_SRC/hls_verify_multiple_exit_tb.vhd
project addfile ../VHDL_SRC/tb_join.vhd
project addfile ../VHDL_SRC/mux.vhd
project addfile ../VHDL_SRC/multiple_exit_wrapper.vhd
project addfile ../VHDL_SRC/fork_dataless.vhd
project addfile ../VHDL_SRC/handshake_constant_2.vhd
project addfile ../VHDL_SRC/handshake_cmpi_1.vhd
project calculateorder
project compileall
eval vsim multiple_exit_wrapper_tb
log -r *
run -all
exit
