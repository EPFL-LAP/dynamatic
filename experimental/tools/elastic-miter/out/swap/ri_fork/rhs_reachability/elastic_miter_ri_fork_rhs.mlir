module {
  handshake.func @ri_fork_rhs(%arg0: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %5 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %2 = ndwire %7 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_C_out"} : <i1>
    %3:2 = fork [2] %0 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    %4 = buffer %3#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer1"} : <i1>
    %5 = spec_v2_repeating_init %4 {handshake.bb = 1 : ui32, handshake.name = "ri1", initToken = 1 : ui1} : <i1>
    %6 = buffer %3#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 1 : ui32, handshake.name = "buffer2"} : <i1>
    %7 = spec_v2_repeating_init %6 {handshake.bb = 1 : ui32, handshake.name = "ri2", initToken = 1 : ui1} : <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %1, %2 : <i1>, <i1>
  }
}
