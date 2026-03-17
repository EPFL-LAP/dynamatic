// RUN: %export-vhdl
// RUN: FileCheck %s -input-file %t/handshake_cmpf_0.vhd

module {
  hw.module @test(in %var0 : !handshake.channel<i32>, in %var3 : !handshake.channel<i32>, in %start : !handshake.control<>, in %clk : i1, in %rst : i1, out out0 : !handshake.channel<i1>, out end : !handshake.control<>) {
    %cmpf0.result = hw.instance "cmpf0" @handshake_cmpf_0(lhs: %var0: !handshake.channel<i32>, rhs: %var3: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %cmpf0.result, %start : !handshake.channel<i1>, !handshake.control<>
  }

  // CHECK-LABEL: architecture {{.*}} of handshake_cmpf_0
  // CHECK: result(0) <= not unordered;
  hw.module.extern @handshake_cmpf_0(in %lhs : !handshake.channel<i32>, in %rhs : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.cmpf", hw.parameters = {DATA_TYPE = !handshake.channel<f32>, FPU_IMPL = "flopoco", INTERNAL_DELAY = "0_000000", LATENCY = 0 : ui32, PREDICATE = "ord"}}
}
