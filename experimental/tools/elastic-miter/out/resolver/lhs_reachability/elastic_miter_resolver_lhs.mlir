module {
  handshake.func @resolver_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["Actual", "Generated"], resNames = ["Confirm"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Actual"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Generated"} : <i1>
    %2 = ndwire %8#1 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_Confirm"} : <i1>
    %3 = passer %0[%8#0] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    %4 = buffer %3 {handshake.bb = 1 : ui32, handshake.name = "buf1", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %5 = buffer %4 {handshake.bb = 1 : ui32, handshake.name = "buf2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    %6 = spec_v2_repeating_init %5 {handshake.bb = 1 : ui32, handshake.name = "ri"} : <i1>
    %7 = spec_v2_interpolator %6, %1 {handshake.bb = 1 : ui32, handshake.name = "interpolator"} : <i1>
    %8:2 = fork [2] %7 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %2 : <i1>
  }
}
