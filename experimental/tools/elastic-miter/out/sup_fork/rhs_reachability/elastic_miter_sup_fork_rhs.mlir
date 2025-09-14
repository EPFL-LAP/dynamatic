module {
  handshake.func @sup_fork_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in", "Cond"], resNames = ["B_out", "C_out"]} {
    %0:2 = fork [2] %arg0 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    %1:2 = fork [2] %arg1 {handshake.bb = 1 : ui32, handshake.name = "fork_cond"} : <i1>
    %2 = passer %0#0[%1#0] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %3 = passer %0#1[%1#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %2, %3 : <i1>, <i1>
  }
}
