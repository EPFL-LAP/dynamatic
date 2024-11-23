module {
  hw.module @a_lhs(in %D : !handshake.channel<i32>, in %C : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out T : !handshake.channel<i32>, out F : !handshake.channel<i32>) {
    %branch.trueOut, %branch.falseOut = hw.instance "branch" @handshake_cond_br_0(condition: %C: !handshake.channel<i1>, data: %D: !handshake.channel<i32>, clk: %clk: i1, rst: %rst: i1) -> (trueOut: !handshake.channel<i32>, falseOut: !handshake.channel<i32>)
    hw.output %branch.trueOut, %branch.falseOut : !handshake.channel<i32>, !handshake.channel<i32>
  }
  hw.module.extern @handshake_cond_br_0(in %condition : !handshake.channel<i1>, in %data : !handshake.channel<i32>, in %clk : i1, in %rst : i1, out trueOut : !handshake.channel<i32>, out falseOut : !handshake.channel<i32>) attributes {hw.name = "handshake.cond_br", hw.parameters = {DATA_TYPE = !handshake.channel<i32>}}
  hw.module @a_lhs_wrapper(in %D : !handshake.channel<i32>, in %C : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out T : !handshake.channel<i32>, out F : !handshake.channel<i32>) {
    %a_lhs_wrapped.T, %a_lhs_wrapped.F = hw.instance "a_lhs_wrapped" @a_lhs(D: %D: !handshake.channel<i32>, C: %C: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (T: !handshake.channel<i32>, F: !handshake.channel<i32>)
    hw.output %a_lhs_wrapped.T, %a_lhs_wrapped.F : !handshake.channel<i32>, !handshake.channel<i32>
  }
}

