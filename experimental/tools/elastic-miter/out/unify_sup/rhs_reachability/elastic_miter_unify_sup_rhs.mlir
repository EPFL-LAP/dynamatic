module {
  handshake.func @unify_sup_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "Cond1", "Cond2"], resNames = ["B_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Cond1"} : <i1>
    %2 = ndwire %arg2 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Cond2"} : <i1>
    %3 = ndwire %5 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %4 = andi %1, %2 {handshake.bb = 1 : ui32, handshake.name = "andi"} : <i1>
    %5 = passer %0[%4] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %3 : <i1>
  }
}
