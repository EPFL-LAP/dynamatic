module {
  handshake.func @suppressorInduction_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["short", "A_in"], resNames = ["B_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_short"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %2 = ndwire %10 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %3:2 = fork [2] %0 {handshake.bb = 1 : ui32, handshake.name = "fork_short"} : <i1>
    %4 = spec_v2_nd_speculator %3#0 {handshake.bb = 1 : ui32, handshake.name = "nd_spec"} : <i1>
    %5:2 = fork [2] %4 {handshake.bb = 2 : ui32, handshake.name = "vm_fork_1"} : <i1>
    %6 = spec_v2_interpolator %3#1, %5#0 {handshake.bb = 2 : ui32, handshake.name = "interpolate"} : <i1>
    %7 = buffer %5#1, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.bb = 2 : ui32, handshake.name = "buffer"} : <i1>
    %8 = spec_v2_repeating_init %7 {handshake.bb = 2 : ui32, handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %9 = passer %1[%8] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %10 = passer %9[%6] {handshake.bb = 2 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %2 : <i1>
  }
}
