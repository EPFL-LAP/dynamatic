module {
  handshake.func @ri_fork_lhs(%arg0: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %5#0 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %2 = ndwire %5#1 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_C_out"} : <i1>
    %3 = buffer %0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer"} : <i1>
    %4 = spec_v2_repeating_init %3 {handshake.bb = 1 : ui32, handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %5:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %1, %2 : <i1>, <i1>
  }
}
