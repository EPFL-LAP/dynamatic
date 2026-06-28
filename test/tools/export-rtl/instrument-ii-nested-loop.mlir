// RUN: %export-vhdl --instrument-ii
// RUN: FileCheck %s --input-file %t/nested.vhd --check-prefix=VHDL
// RUN: %export-verilog --instrument-ii
// RUN: FileCheck %s --input-file %t/nested.v --check-prefix=VERILOG

// Tests that `export-rtl --instrument-ii` instruments only the *innermost* loop
// of a loop nest, in both VHDL and Verilog. This kernel has two loop headers:
// `control_merge0` (outer) and `control_merge2` (inner). Only the inner loop
// must get a monitor, even though data flows from the outer loop into the inner
// one. Iterations of the inner loop are counted on its header's control output,
// and the II is reported once when control leaves the inner loop.

// VHDL:      -- II instrumentation for innermost loop "control_merge2"
// VHDL-NEXT: ii_monitor_control_merge2 : process(clk)
// VHDL:      if (control_merge2_outs_valid = '1') and (control_merge2_outs_ready = '1') then
// VHDL:      if (cond_br12_falseOut_valid = '1' and cond_br12_falseOut_ready = '1') then
// VHDL:      report "II_INSTRUMENT: loop control_merge2 average_II=" {{.*}} severity note;
// VHDL:      end process;
// The outer loop (control_merge0) must not be instrumented: exactly one monitor.
// VHDL-NOT: process(clk)
// VHDL-NOT: II_INSTRUMENT

// VERILOG:      // II instrumentation for innermost loop "control_merge2"
// VERILOG:      always @(posedge clk) begin
// VERILOG:      if (control_merge2_outs_valid && control_merge2_outs_ready) begin
// VERILOG:      if ((cond_br12_falseOut_valid && cond_br12_falseOut_ready)) begin
// VERILOG:      $display("II_INSTRUMENT: loop control_merge2 average_II=%0d iterations=%0d"{{.*}});
// VERILOG-NOT: always @(posedge clk)
// VERILOG-NOT: II_INSTRUMENT

module {
  hw.module @nested(in %in0 : !handshake.channel<i32>, in %in1 : !handshake.channel<i32>, in %start : !handshake.control<>, in %clk : i1, in %rst : i1, out end : !handshake.control<>) {
    %fork0.outs_0, %fork0.outs_1, %fork0.outs_2, %fork0.outs_3 = hw.instance "fork0" @handshake_fork_0(ins: %start: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>, outs_2: !handshake.control<>, outs_3: !handshake.control<>)
    %constant0.outs = hw.instance "constant0" @handshake_constant_0(ctrl: %fork0.outs_1: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %constant1.outs = hw.instance "constant1" @handshake_constant_1(ctrl: %fork0.outs_0: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %fork1.outs_0, %fork1.outs_1 = hw.instance "fork1" @handshake_fork_1(ins: %constant0.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %mux0.outs = hw.instance "mux0" @handshake_mux_0(index: %fork4.outs_4: !handshake.channel<i1>, ins_0: %fork1.outs_0: !handshake.channel<i32>, ins_1: %addi1.result: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %fork2.outs_0, %fork2.outs_1 = hw.instance "fork2" @handshake_fork_1(ins: %mux0.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %mux1.outs = hw.instance "mux1" @handshake_mux_0(index: %fork4.outs_3: !handshake.channel<i1>, ins_0: %in0: !handshake.channel<i32>, ins_1: %cond_br6.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %fork3.outs_0, %fork3.outs_1 = hw.instance "fork3" @handshake_fork_1(ins: %mux1.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %mux2.outs = hw.instance "mux2" @handshake_mux_0(index: %fork4.outs_2: !handshake.channel<i1>, ins_0: %in1: !handshake.channel<i32>, ins_1: %cond_br7.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %mux3.outs = hw.instance "mux3" @handshake_mux_0(index: %fork4.outs_1: !handshake.channel<i1>, ins_0: %fork1.outs_1: !handshake.channel<i32>, ins_1: %cond_br8.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %mux4.outs = hw.instance "mux4" @handshake_mux_0(index: %fork4.outs_0: !handshake.channel<i1>, ins_0: %constant1.outs: !handshake.channel<i32>, ins_1: %fork12.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %control_merge0.outs, %control_merge0.index = hw.instance "control_merge0" @handshake_control_merge_0(ins_0: %fork0.outs_3: !handshake.control<>, ins_1: %cond_br12.falseOut: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>, index: !handshake.channel<i1>)
    %fork4.outs_0, %fork4.outs_1, %fork4.outs_2, %fork4.outs_3, %fork4.outs_4 = hw.instance "fork4" @handshake_fork_2(ins: %control_merge0.index: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>, outs_3: !handshake.channel<i1>, outs_4: !handshake.channel<i1>)
    %cmpi0.result = hw.instance "cmpi0" @handshake_cmpi_0(lhs: %fork2.outs_0: !handshake.channel<i32>, rhs: %fork3.outs_0: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %fork5.outs_0, %fork5.outs_1, %fork5.outs_2, %fork5.outs_3, %fork5.outs_4, %fork5.outs_5 = hw.instance "fork5" @handshake_fork_3(ins: %cmpi0.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>, outs_3: !handshake.channel<i1>, outs_4: !handshake.channel<i1>, outs_5: !handshake.channel<i1>)
    %cond_br0.trueOut, %cond_br0.falseOut = hw.instance "cond_br0" @handshake_cond_br_0(condition: %fork5.outs_0: !handshake.channel<i1>, data: %fork3.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "sink0" @handshake_sink_0(ins: %cond_br0.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %cond_br1.trueOut, %cond_br1.falseOut = hw.instance "cond_br1" @handshake_cond_br_0(condition: %fork5.outs_1: !handshake.channel<i1>, data: %mux2.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "sink1" @handshake_sink_0(ins: %cond_br1.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %cond_br2.trueOut, %cond_br2.falseOut = hw.instance "cond_br2" @handshake_cond_br_0(condition: %fork5.outs_2: !handshake.channel<i1>, data: %mux3.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "sink2" @handshake_sink_0(ins: %cond_br2.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %cond_br3.trueOut, %cond_br3.falseOut = hw.instance "cond_br3" @handshake_cond_br_0(condition: %fork5.outs_3: !handshake.channel<i1>, data: %mux4.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "sink3" @handshake_sink_0(ins: %cond_br3.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %cond_br4.trueOut, %cond_br4.falseOut = hw.instance "cond_br4" @handshake_cond_br_0(condition: %fork5.outs_4: !handshake.channel<i1>, data: %fork2.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "sink4" @handshake_sink_0(ins: %cond_br4.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %cond_br5.trueOut, %cond_br5.falseOut = hw.instance "cond_br5" @handshake_cond_br_1(condition: %fork5.outs_5: !handshake.channel<i1>, data: %control_merge0.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.control<>, falseOut: !handshake.control<>)
    %fork6.outs_0, %fork6.outs_1 = hw.instance "fork6" @handshake_fork_1(ins: %cond_br2.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %mux5.outs = hw.instance "mux5" @handshake_mux_0(index: %fork9.outs_5: !handshake.channel<i1>, ins_0: %fork6.outs_0: !handshake.channel<i32>, ins_1: %addi0.result: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %fork7.outs_0, %fork7.outs_1 = hw.instance "fork7" @handshake_fork_1(ins: %mux5.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %mux6.outs = hw.instance "mux6" @handshake_mux_0(index: %fork9.outs_4: !handshake.channel<i1>, ins_0: %cond_br0.trueOut: !handshake.channel<i32>, ins_1: %cond_br6.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %mux7.outs = hw.instance "mux7" @handshake_mux_0(index: %fork9.outs_3: !handshake.channel<i1>, ins_0: %cond_br1.trueOut: !handshake.channel<i32>, ins_1: %cond_br7.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %fork8.outs_0, %fork8.outs_1 = hw.instance "fork8" @handshake_fork_1(ins: %mux7.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %mux8.outs = hw.instance "mux8" @handshake_mux_0(index: %fork9.outs_2: !handshake.channel<i1>, ins_0: %fork6.outs_1: !handshake.channel<i32>, ins_1: %cond_br8.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %mux9.outs = hw.instance "mux9" @handshake_mux_0(index: %fork9.outs_1: !handshake.channel<i1>, ins_0: %cond_br3.trueOut: !handshake.channel<i32>, ins_1: %fork11.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %mux10.outs = hw.instance "mux10" @handshake_mux_0(index: %fork9.outs_0: !handshake.channel<i1>, ins_0: %cond_br4.trueOut: !handshake.channel<i32>, ins_1: %cond_br10.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %control_merge2.outs, %control_merge2.index = hw.instance "control_merge2" @handshake_control_merge_0(ins_0: %cond_br5.trueOut: !handshake.control<>, ins_1: %cond_br12.trueOut: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>, index: !handshake.channel<i1>)
    %fork9.outs_0, %fork9.outs_1, %fork9.outs_2, %fork9.outs_3, %fork9.outs_4, %fork9.outs_5 = hw.instance "fork9" @handshake_fork_3(ins: %control_merge2.index: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>, outs_3: !handshake.channel<i1>, outs_4: !handshake.channel<i1>, outs_5: !handshake.channel<i1>)
    %cmpi1.result = hw.instance "cmpi1" @handshake_cmpi_0(lhs: %fork7.outs_0: !handshake.channel<i32>, rhs: %fork8.outs_0: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %fork10.outs_0, %fork10.outs_1, %fork10.outs_2, %fork10.outs_3, %fork10.outs_4, %fork10.outs_5, %fork10.outs_6 = hw.instance "fork10" @handshake_fork_4(ins: %cmpi1.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>, outs_3: !handshake.channel<i1>, outs_4: !handshake.channel<i1>, outs_5: !handshake.channel<i1>, outs_6: !handshake.channel<i1>)
    %cond_br6.trueOut, %cond_br6.falseOut = hw.instance "cond_br6" @handshake_cond_br_0(condition: %fork10.outs_0: !handshake.channel<i1>, data: %mux6.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    %cond_br7.trueOut, %cond_br7.falseOut = hw.instance "cond_br7" @handshake_cond_br_0(condition: %fork10.outs_1: !handshake.channel<i1>, data: %fork8.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    %cond_br8.trueOut, %cond_br8.falseOut = hw.instance "cond_br8" @handshake_cond_br_0(condition: %fork10.outs_2: !handshake.channel<i1>, data: %mux8.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    %cond_br9.trueOut, %cond_br9.falseOut = hw.instance "cond_br9" @handshake_cond_br_0(condition: %fork10.outs_3: !handshake.channel<i1>, data: %mux9.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    %cond_br10.trueOut, %cond_br10.falseOut = hw.instance "cond_br10" @handshake_cond_br_0(condition: %fork10.outs_4: !handshake.channel<i1>, data: %mux10.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    %cond_br11.trueOut, %cond_br11.falseOut = hw.instance "cond_br11" @handshake_cond_br_0(condition: %fork10.outs_5: !handshake.channel<i1>, data: %fork7.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "sink6" @handshake_sink_0(ins: %cond_br11.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %cond_br12.trueOut, %cond_br12.falseOut = hw.instance "cond_br12" @handshake_cond_br_1(condition: %fork10.outs_6: !handshake.channel<i1>, data: %control_merge2.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.control<>, falseOut: !handshake.control<>)
    %fork11.outs_0, %fork11.outs_1 = hw.instance "fork11" @handshake_fork_1(ins: %cond_br9.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %addi0.result = hw.instance "addi0" @handshake_addi_0(lhs: %cond_br11.trueOut: !handshake.channel<i32>, rhs: %fork11.outs_0: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i32>)
    %fork12.outs_0, %fork12.outs_1 = hw.instance "fork12" @handshake_fork_1(ins: %cond_br9.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %addi1.result = hw.instance "addi1" @handshake_addi_0(lhs: %cond_br10.falseOut: !handshake.channel<i32>, rhs: %fork12.outs_0: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i32>)
    hw.instance "sink10" @handshake_sink_1(ins: %cond_br5.falseOut: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output %fork0.outs_2 : !handshake.control<>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.control<>, out outs_1 : !handshake.control<>, out outs_2 : !handshake.control<>, out outs_3 : !handshake.control<>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.control<>, SIZE = 4 : ui32}}
  hw.module.extern @handshake_constant_0(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i32>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 32 : ui32, VALUE = "00000000000000000000000000000000"}}
  hw.module.extern @handshake_constant_1(in %ctrl : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i32>) attributes {hw.name = "handshake.constant", hw.parameters = {DATA_WIDTH = 32 : ui32, VALUE = "00000000000000000000000000000001"}}
  hw.module.extern @handshake_fork_1(in %ins : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i32>, out outs_1 : !handshake.channel<i32>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i32>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i32>, in %ins_1 : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i32>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i32>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_control_merge_0(in %ins_0 : !handshake.control<>, in %ins_1 : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>, out index : !handshake.channel<i1>) attributes {hw.name = "handshake.control_merge", hw.parameters = {DATA_TYPE = !handshake.control<>, INDEX_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_fork_2(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>, out outs_3 : !handshake.channel<i1>, out outs_4 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 5 : ui32}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i32>, in %rhs : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i32>, PREDICATE = "slt"}}
  hw.module.extern @handshake_fork_3(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>, out outs_3 : !handshake.channel<i1>, out outs_4 : !handshake.channel<i1>, out outs_5 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 6 : ui32}}
  hw.module.extern @handshake_cond_br_0(in %condition : !handshake.channel<i1>, in %data : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out trueOut : !handshake.channel<i32>, out falseOut : !handshake.channel<i32>) attributes {hw.name = "handshake.cond_br", hw.parameters = {DATA_TYPE = !handshake.channel<i32>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i32>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i32>}}
  hw.module.extern @handshake_cond_br_1(in %condition : !handshake.channel<i1>, in %data : !handshake.control<>, in %clk : i1, in %rst : i1, out trueOut : !handshake.control<>, out falseOut : !handshake.control<>) attributes {hw.name = "handshake.cond_br", hw.parameters = {DATA_TYPE = !handshake.control<>}}
  hw.module.extern @handshake_fork_4(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>, out outs_3 : !handshake.channel<i1>, out outs_4 : !handshake.channel<i1>, out outs_5 : !handshake.channel<i1>, out outs_6 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 7 : ui32}}
  hw.module.extern @handshake_addi_0(in %lhs : !handshake.channel<i32>, in %rhs : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i32>) attributes {hw.name = "handshake.addi", hw.parameters = {DATA_TYPE = !handshake.channel<i32>}}
  hw.module.extern @handshake_sink_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.control<>}}
}
