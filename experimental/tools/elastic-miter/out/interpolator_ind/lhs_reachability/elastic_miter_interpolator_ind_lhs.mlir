module {
  handshake.func @interpolator_ind_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in", "Sup_in"], resNames = ["Sup_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_B_in"} : <i1>
    %2 = ndwire %arg2 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Sup_in"} : <i1>
    %3 = ndwire %8 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_Sup_out"} : <i1>
    %4:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    %5 = spec_v2_interpolator %0, %4#0 {handshake.bb = 1 : ui32, handshake.name = "interpolate"} : <i1>
    %6 = spec_v2_repeating_init %4#1 {handshake.bb = 1 : ui32, handshake.name = "ri"} : <i1>
    %7 = passer %2[%6] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %8 = passer %7[%5] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %3 : <i1>
  }
}
