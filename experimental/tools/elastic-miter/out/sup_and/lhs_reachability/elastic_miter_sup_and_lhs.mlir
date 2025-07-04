module {
  handshake.func @sup_and_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in", "Cond"], resNames = ["C_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_B_in"} : <i1>
    %2 = ndwire %arg2 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Cond"} : <i1>
    %3 = ndwire %7 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_C_out"} : <i1>
    %4:2 = fork [2] %2 {handshake.bb = 1 : ui32, handshake.name = "fork_cond"} : <i1>
    %5 = passer %0[%4#0] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %6 = passer %1[%4#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    %7 = andi %5, %6 {handshake.bb = 1 : ui32, handshake.name = "andi"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %3 : <i1>
  }
}
