module {
  hw.module @sup_load_lhs(in %Addr_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Data_out : !handshake.channel<i1>) {
    %ndw_in_Addr_in.outs = hw.instance "ndw_in_Addr_in" @handshake_ndwire_0(ins: %Addr_in: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_in_Cond.outs = hw.instance "ndw_in_Cond" @handshake_ndwire_0(ins: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %ndw_out_Data_out.outs = hw.instance "ndw_out_Data_out" @handshake_ndwire_0(ins: %load0.dataOut: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    %passer.result = hw.instance "passer" @handshake_passer_0(data: %ndw_in_Addr_in.outs: !handshake.channel<i1>, ctrl: %ndw_in_Cond.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (result: !handshake.channel<i1>)
    %load0.addrOut, %load0.dataOut = hw.instance "load0" @handshake_load_0(addrIn: %passer.result: !handshake.channel<i1>, dataFromMem: %mc_buffer.outs: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (addrOut: !handshake.channel<i1>, dataOut: !handshake.channel<i1>)
    %mc_buffer.outs = hw.instance "mc_buffer" @handshake_buffer_0(ins: %load0.addrOut: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (outs: !handshake.channel<i1>)
    hw.output %ndw_out_Data_out.outs : !handshake.channel<i1>
  }
  hw.module.extern @handshake_ndwire_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.ndwire", hw.parameters = {DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_passer_0(in %data : !handshake.channel<i1>, in %ctrl : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out result : !handshake.channel<i1>) attributes {hw.name = "handshake.passer", hw.parameters = {}}
  hw.module.extern @handshake_load_0(in %addrIn : !handshake.channel<i1>, in %dataFromMem : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out addrOut : !handshake.channel<i1>, out dataOut : !handshake.channel<i1>) attributes {hw.name = "handshake.load", hw.parameters = {ADDR_TYPE = !handshake.channel<i1>, DATA_TYPE = !handshake.channel<i1>}}
  hw.module.extern @handshake_buffer_0(in %ins : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out outs : !handshake.channel<i1>) attributes {hw.name = "handshake.buffer", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", DATA_TYPE = !handshake.channel<i1>, DEBUG_COUNTER = false, NUM_SLOTS = 1 : ui32}}
  hw.module @sup_load_lhs_wrapper(in %Addr_in : !handshake.channel<i1>, in %Cond : !handshake.channel<i1>, in %clk : i1, in %rst : i1, out Data_out : !handshake.channel<i1>) {
    %sup_load_lhs_wrapped.Data_out = hw.instance "sup_load_lhs_wrapped" @sup_load_lhs(Addr_in: %Addr_in: !handshake.channel<i1>, Cond: %Cond: !handshake.channel<i1>, clk: %clk: i1, rst: %rst: i1) -> (Data_out: !handshake.channel<i1>)
    hw.output %sup_load_lhs_wrapped.Data_out : !handshake.channel<i1>
  }
}

