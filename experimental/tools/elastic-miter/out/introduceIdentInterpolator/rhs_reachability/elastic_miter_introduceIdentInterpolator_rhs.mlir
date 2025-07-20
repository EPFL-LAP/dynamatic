module {
  handshake.func @introduceIdentInterpolator_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "val"], resNames = ["B_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_val"} : <i1>
    %2 = ndwire %5 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %3:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "vm_fork_1"} : <i1>
    %4 = spec_v2_interpolator %3#0, %3#1 {handshake.bb = 1 : ui32, handshake.name = "interpolate"} : <i1>
    %5 = passer %0[%4] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %2 : <i1>
  }
}
