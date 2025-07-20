module {
  handshake.func @introduceIdentInterpolator_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "val"], resNames = ["B_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_val"} : <i1>
    %2 = ndwire %3 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %3 = passer %0[%1] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %2 : <i1>
  }
}
