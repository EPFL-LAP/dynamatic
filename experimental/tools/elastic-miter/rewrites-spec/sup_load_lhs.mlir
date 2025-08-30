module {
  handshake.func @sup_load_lhs(%addr: !handshake.channel<i1>, %cond: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["Addr_in", "Cond"], resNames = ["Data_out"]} {
    %addr_passer = passer %addr [%cond] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    %addrResult, %dataResult = load[%addr_passer] %data {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i1>, <i1>, <i1>, <i1>
    %data = buffer %addrResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "mc_buffer", debugCounter = false} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %dataResult : <i1>
  }
}
