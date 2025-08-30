module {
  handshake.func @sup_load_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["Addr_in", "Cond"], resNames = ["Data_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Addr_in"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Cond"} : <i1>
    %2 = ndwire %4 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_Data_out"} : <i1>
    %addressResult, %dataResult = load[%0] %3 {handshake.bb = 2 : ui32, handshake.bufProps = #handshake<bufProps{"1": [0,0], [0,0], 0, 0.000000e+00, 0.000000e+00, 0.000000e+00}>, handshake.mem_interface = #handshake.mem_interface<MC>, handshake.name = "load0"} : <i1>, <i1>, <i1>, <i1>
    %3 = buffer %addressResult, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.bb = 2 : ui32, handshake.name = "mc_buffer"} : <i1>
    %4 = passer %dataResult[%1] {handshake.bb = 2 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %2 : <i1>
  }
}
