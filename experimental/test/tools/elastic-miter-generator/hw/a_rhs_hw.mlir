module {
  hw.module @a_rhs(in %D : !handshake.channel<i32>, in %C : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out T : !handshake.channel<i32>, out F : !handshake.channel<i32>) {
    %fork_data.outs_0, %fork_data.outs_1 = hw.instance "fork_data" @handshake_fork_0(ins: %D: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i32>, outs_1: !handshake.channel<i32>)
    %fork_control.outs_0, %fork_control.outs_1 = hw.instance "fork_control" @handshake_fork_1(ins: %C: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %not.outs = hw.instance "not" @handshake_not_0(ins: %fork_control.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %supp_br_0.trueOut, %supp_br_0.falseOut = hw.instance "supp_br_0" @handshake_cond_br_0(condition: %not.outs: !handshake.channel<i1>, data: %fork_data.outs_0: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "supp_sink_0" @handshake_sink_0(ins: %supp_br_0.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    %supp_br_1.trueOut, %supp_br_1.falseOut = hw.instance "supp_br_1" @handshake_cond_br_0(condition: %fork_control.outs_1: !handshake.channel<i1>, data: %fork_data.outs_1: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.instance "supp_sink_1" @handshake_sink_0(ins: %supp_br_1.trueOut: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> ()
    hw.output %supp_br_0.falseOut, %supp_br_1.falseOut : !handshake.channel<i32>, !handshake.channel<i32>
  }
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i32>, out outs_1 : !handshake.channel<i32>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i32>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_fork_1(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_cond_br_0(in %condition : !handshake.channel<i1>, in %data : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out trueOut : !handshake.channel<i32>, out falseOut : !handshake.channel<i32>) attributes {hw.name = "handshake.cond_br", hw.parameters = {DATA_TYPE = !handshake.channel<i32>}}
  hw.module.extern @handshake_sink_0(in %ins : !handshake.channel<i32>, in %clk : i1, in %rst : i1) attributes {hw.name = "handshake.sink", hw.parameters = {DATA_TYPE = !handshake.channel<i32>}}
  hw.module @a_rhs_wrapper(in %D : !handshake.channel<i32>, in %C : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out T : !handshake.channel<i32>, out F : !handshake.channel<i32>) {
    %a_rhs_wrapped.T, %a_rhs_wrapped.F = hw.instance "a_rhs_wrapped" @a_rhs(D: %D: !handshake.channel<i32>, C: %C: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (T: !handshake.channel<i32>, F: !handshake.channel<i32>)
    hw.output %a_rhs_wrapped.T, %a_rhs_wrapped.F : !handshake.channel<i32>, !handshake.channel<i32>
  }
}

