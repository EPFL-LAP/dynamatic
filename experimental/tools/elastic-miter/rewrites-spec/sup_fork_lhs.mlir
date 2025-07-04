module {
  handshake.func @sup_fork_lhs(%a: !handshake.channel<i1>, %cond: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in", "Cond"], resNames = ["B_out", "C_out"]} {
    %a_passer = passer %a [%cond] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %result:2 = fork [2] %a_passer {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %result#0, %result#1 : <i1>, <i1>
  }
}
