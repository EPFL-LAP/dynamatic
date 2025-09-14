module {
  handshake.func @sup_fork_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in", "Cond"], resNames = ["B_out", "C_out"]} {
    %0 = passer %arg0[%arg1] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %1:2 = fork [2] %0 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %1#0, %1#1 : <i1>, <i1>
  }
}
