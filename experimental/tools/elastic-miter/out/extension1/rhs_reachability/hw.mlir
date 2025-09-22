module {
  hw.module @extension1_rhs(in %arg : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %initop.outs = hw.instance "initop" @handshake_init_0(ins: %buff.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %cst_dup_fork.outs_0, %cst_dup_fork.outs_1, %cst_dup_fork.outs_2 = hw.instance "cst_dup_fork" @handshake_fork_0(ins: %initop.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>, outs_2: !handshake.channel<i1>)
    %not.outs = hw.instance "not" @handshake_not_0(ins: %cst_dup_fork.outs_2: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %buff.outs = hw.instance "buff" @handshake_buffer_0(ins: %not.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %arg_fork.outs_0, %arg_fork.outs_1 = hw.instance "arg_fork" @handshake_fork_1(ins: %arg: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs_0: !handshake.channel<i1>, outs_1: !handshake.channel<i1>)
    %arg_not.outs = hw.instance "arg_not" @handshake_not_0(ins: %arg_fork.outs_0: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %arg_mux.outs = hw.instance "arg_mux" @handshake_mux_0(index: %cst_dup_fork.outs_0: !handshake.channel<i1>, ins_0: %arg_not.outs: !handshake.channel<i1>, ins_1: %arg_fork.outs_1: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %cst_passer.result = hw.instance "cst_passer" @handshake_passer_0(data: %cst_dup_fork.outs_1: !handshake.channel<i1>, ctrl: %arg_mux.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    hw.output %cst_passer.result : !handshake.channel<i1>
  }
  hw.module.extern @handshake_init_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.init", hw.parameters = {INIT_TOKEN = 0 : ui32}}
  hw.module.extern @handshake_fork_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>, out outs_2 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 3 : ui32}}
  hw.module.extern @handshake_not_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.not", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, DEBUG_COUNTER = false, NUM_SLOTS = 1 : ui32}}
  hw.module.extern @handshake_fork_1(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs_0 : !handshake.channel<i1>, out outs_1 : !handshake.channel<i1>) attributes {hw.name = "handshake.fork", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_mux_0(in %index : !handshake.channel<i1>, in %ins_0 : !handshake.channel<i1>, in %ins_1 : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.mux", hw.parameters = {DATA_TYPE = !handshake.channel<i1>, SELECT_TYPE = !handshake.channel<i1>, SIZE = 2 : ui32}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module @extension1_rhs_wrapper(in %arg : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out res : !handshake.channel<i1>) {
    %extension1_rhs_wrapped.res = hw.instance "extension1_rhs_wrapped" @extension1_rhs(arg: %arg: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (res: !handshake.channel<i1>)
    hw.output %extension1_rhs_wrapped.res : !handshake.channel<i1>
  }
}

