vlib work
vmap work work
project new . simulation work modelsim.ini 0
project open simulation
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/tb_join.vhd
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/two_port_RAM.vhd
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/single_argument.vhd
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/simpackage.vhd
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/tb_minimal.vhd
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/minimal.v
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/minimal_wrapper.v
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/one_slot_break_dvr.v
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/shift_reg_break_dv.v
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/tfifo_dataless.v
project addfile /home/iaganz/devian/dynamatic/experimental/tools/unit-generators/verilog/test_modules/minimal/out/sim/HDL_SRC/tfifo.v
project calculateorder
project compileall
eval vsim tb
log -r *
run -all
exit
