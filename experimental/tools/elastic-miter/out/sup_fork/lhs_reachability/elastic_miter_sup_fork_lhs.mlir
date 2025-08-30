module {
  handshake.func @sup_fork_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in", "Cond"], resNames = ["B_out", "C_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Cond"} : <i1>
    %2 = ndwire %5#0 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %3 = ndwire %5#1 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_C_out"} : <i1>
    %4 = passer %0[%1] {handshake.bb = 2 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %5:2 = fork [2] %4 {handshake.bb = 2 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end0"} %2, %3 : <i1>, <i1>
  }
}
