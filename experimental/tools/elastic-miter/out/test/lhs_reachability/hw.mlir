module {
  hw.module @test_lhs(in %A_in : !handshake.control<>, in %clk : i1, in %rst : i1) {
    %ndw_in_A_in.outs = hw.instance "ndw_in_A_in" @handshake_ndwire_0(ins: %A_in: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    %nds.result = hw.instance "nds" @handshake_ndsource_0(clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.control<>)
    %join.outs = hw.instance "join" @handshake_join_0(ins_0: %ndw_in_A_in.outs: !handshake.control<>, ins_1: %nds.result: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.control<>)
    hw.instance "sink" @handshake_sink_0(ins: %join.outs: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.control<>}}
  hw.module.extern @handshake_ndsource_0(in %clk : i1, in %rst : i1, out result : !handshake.control<>) attributes {hw.name = "handshake.ndsource", hw.parameters = {}}
  hw.module.extern @handshake_join_0(in %ins_0 : !handshake.control<>, in %ins_1 : !handshake.control<>, in %clk : i1, in %rst : i1, out outs : !handshake.control<>) attributes {hw.name = "handshake.join", hw.parameters = {SIZE = 2 : ui32}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.control<>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.control<>}}
  hw.module @test_lhs_wrapper(in %A_in : !handshake.control<>, in %clk : i1, in %rst : i1) {
    hw.instance "test_lhs_wrapped" @test_lhs(A_in: %A_in: !handshake.control<>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output
  }
}

