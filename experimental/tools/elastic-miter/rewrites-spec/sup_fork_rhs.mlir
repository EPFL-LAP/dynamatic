module {
  handshake.func @sup_fork_rhs(%a: !handshake.channel<i1>, %cond: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in", "Cond"], resNames = ["B_out", "C_out"]} {
    %result:2 = fork [2] %a {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    %cond_forked:2 = fork [2] %cond {handshake.bb = 1 : ui32, handshake.name = "fork_cond"} : <i1>
    %b_passer = passer %result#0 [%cond_forked#0] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %c_passer = passer %result#1 [%cond_forked#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %b_passer, %c_passer : <i1>, <i1>
  }
}
