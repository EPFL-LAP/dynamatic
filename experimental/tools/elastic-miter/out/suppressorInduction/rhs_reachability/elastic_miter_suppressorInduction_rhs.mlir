module {
  handshake.func @suppressorInduction_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["short", "A_in"], resNames = ["B_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_short"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %2 = ndwire %8 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %3:2 = fork [2] %0 {handshake.bb = 1 : ui32, handshake.name = "fork_short"} : <i1>
    %4 = spec_v2_nd_speculator %3#0 {handshake.bb = 1 : ui32, handshake.name = "nd_spec"} : <i1>
    %5 = spec_v2_repeating_init %4 {handshake.bb = 2 : ui32, handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %6 = buffer %5, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer"} : <i1>
    %7 = spec_v2_interpolator %3#1, %6 {handshake.bb = 2 : ui32, handshake.name = "interpolate"} : <i1>
    %8 = passer %1[%7] {handshake.bb = 2 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"} %2 : <i1>
  }
}
