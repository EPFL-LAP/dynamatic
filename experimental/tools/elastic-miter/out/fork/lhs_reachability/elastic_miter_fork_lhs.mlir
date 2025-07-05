module {
  handshake.func @fork_lhs(%arg0: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %3#0 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %2 = ndwire %3#1 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_C_out"} : <i1>
    %3:2 = fork [2] %0 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %1, %2 : <i1>, <i1>
  }
}
