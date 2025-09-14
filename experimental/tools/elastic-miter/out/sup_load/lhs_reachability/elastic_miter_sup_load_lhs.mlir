module {
  handshake.func @sup_load_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["Addr_in", "Cond"], resNames = ["Data_out"]} {
    %0 = passer %arg0[%arg1] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    %addressResult, %dataResult = load[%0] %1 {handshake.bb = 1 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i1>, <i1>, <i1>, <i1>
    %1 = buffer %addressResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 1 : ui32, handshake.name = "mc_buffer"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %dataResult : <i1>
  }
}
