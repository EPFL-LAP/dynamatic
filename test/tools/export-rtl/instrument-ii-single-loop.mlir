// RUN: %export-vhdl --instrument-ii
// RUN: FileCheck %s --input-file %t/selfLoop.vhd --check-prefix=VHDL
// RUN: %export-verilog --instrument-ii
// RUN: FileCheck %s --input-file %t/selfLoop.v --check-prefix=VERILOG

// Tests that `export-rtl --instrument-ii` detects the loop of a single-loop
// kernel (its control-merge header is `control_merge0`) and emits a monitor
// (process/always block) that measures the average initiation interval (II),
// in both VHDL and Verilog.

// Iterations are counted on the header's control output; the II is reported
// once, when control leaves the loop (the loop-exit branch), rather than on
// every iteration.

// VHDL:      -- II instrumentation for innermost loop "control_merge0"
// VHDL-NEXT: ii_monitor_control_merge0 : process(clk)
// VHDL:      if (control_merge0_outs_valid = '1') and (control_merge0_outs_ready = '1') then
// VHDL:      if (cond_br2_falseOut_valid = '1' and cond_br2_falseOut_ready = '1') then
// VHDL:      report "II_INSTRUMENT: loop control_merge0 average_II=" {{.*}} severity note;
// VHDL:      end process;

// VERILOG:      // II instrumentation for innermost loop "control_merge0"
// VERILOG:      always @(posedge clk) begin
// VERILOG:      if (control_merge0_outs_valid && control_merge0_outs_ready) begin
// VERILOG:      if ((cond_br2_falseOut_valid && cond_br2_falseOut_ready)) begin
// VERILOG:      $display("II_INSTRUMENT: loop control_merge0 average_II=%0d iterations=%0d"{{.*}});

module {
  hw.module @selfLoop(in %in0 : !handshake.channel<i32>, in %in1 : !handshake.channel<i32>, in %start : !handshake.control<>, in %clk : i1, in %rst : i1, out end : !handshake.control<>) {
    %fork0.outs_0, %fork0.outs_1 = hw.instance "fork0" @handshake_fork_0(ins: %start: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.control<>, outs_1: !handshake.control<>)
    %mux0.outs = hw.instance "mux0" @handshake_mux_0(index: %fork3.outs_1: !handshake.channel<i1>, ins_0: %in0: !handshake.channel<i32>, ins_1: %cond_br0.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %fork1.outs_0, %fork1.outs_1 = hw.instance "fork1" @handshake_fork_1(ins: %mux0.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %mux1.outs = hw.instance "mux1" @handshake_mux_0(index: %fork3.outs_0: !handshake.channel<i1>, ins_0: %in1: !handshake.channel<i32>, ins_1: %cond_br1.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i32>)
    %fork2.outs_0, %fork2.outs_1 = hw.instance "fork2" @handshake_fork_1(ins: %mux1.outs: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %control_merge0.outs, %control_merge0.index = hw.instance "control_merge0" @handshake_control_merge_0(ins_0: %fork0.outs_1: !handshake.control<>, ins_1: %cond_br2.trueOut: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>, index: !handshake.channel<i1>)
    %fork3.outs_0, %fork3.outs_1 = hw.instance "fork3" @handshake_fork_2(ins: %control_merge0.index: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %cmpi0.result = hw.instance "cmpi0" @handshake_cmpi_0(lhs: %fork1.outs_0: !handshake.channel<i32>, rhs: %fork2.outs_0: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %fork4.outs_0, %fork4.outs_1, %fork4.outs_2 = hw.instance "fork4" @handshake_fork_3(ins: %cmpi0.result: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %cond_br0.trueOut, %cond_br0.falseOut = hw.instance "cond_br0" @handshake_cond_br_0(condition: %fork4.outs_0: !handshake.channel<i1>, data: %fork1.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "sink0" @handshake_sink_0(ins: %cond_br0.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %cond_br1.trueOut, %cond_br1.falseOut = hw.instance "cond_br1" @handshake_cond_br_0(condition: %fork4.outs_1: !handshake.channel<i1>, data: %fork2.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "sink1" @handshake_sink_0(ins: %cond_br1.falseOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %cond_br2.trueOut, %cond_br2.falseOut = hw.instance "cond_br2" @handshake_cond_br_1(condition: %fork4.outs_2: !handshake.channel<i1>, data: %control_merge0.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.control<>, falseOut: !handshake.control<>)
    hw.instance "sink3" @handshake_sink_1(ins: %cond_br2.falseOut: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output %fork0.outs_0 : !handshake.control<>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.control<>, out outs_1 : !handshake.control<>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.control<>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i32>, in %ins_1 : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i32>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i32>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_fork_1(in %ins : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i32>, out outs_1 : !handshake.channel<i32>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i32>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_control_merge_0(in %ins_0 : !handshake.control<>, in %ins_1 : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>, out index : !handshake.channel<i1>) attributes {hw.name = "handshake.control_merge", hw.parameters = {DATA_TYPE = !handshake.control<>, INDEX_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_fork_2(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_cmpi_0(in %lhs : !handshake.channel<i32>, in %rhs : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpi", hw.parameters = {DATA_TYPE = !handshake.channel<i32>, PREDICATE = "eq"}}
  hw.module.extern @handshake_fork_3(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_cond_br_0(in %condition : !handshake.channel<i1>, in %data : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out trueOut : !handshake.channel<i32>, out falseOut : !handshake.channel<i32>) attributes {hw.name = "handshake.cond_br", hw.parameters = {DATA_TYPE = !handshake.channel<i32>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i32>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i32>}}
  hw.module.extern @handshake_cond_br_1(in %condition : !handshake.channel<i1>, in %data : !handshake.control<>, in %clk : i1, in %rst : i1, out trueOut : !handshake.control<>, out falseOut : !handshake.control<>) attributes {hw.name = "handshake.cond_br", hw.parameters = {DATA_TYPE = !handshake.control<>}}
  hw.module.extern @handshake_sink_1(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.control<>}}
}
