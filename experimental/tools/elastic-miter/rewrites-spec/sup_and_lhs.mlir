module {
  handshake.func @sup_and_lhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %cond: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "B_in", "Cond"], resNames = ["C_out"]} {
    %cond_forked:2 = fork [2] %cond {handshake.bb = 1 : ui32, handshake.name = "fork_cond"} : <i1>
    %a_passer = passer %a [%cond_forked#0] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %b_passer = passer %b [%cond_forked#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    %c = andi %a_passer, %b_passer {handshake.bb = 1 : ui32, handshake.name = "andi"} : <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %c : <i1>
  }
}
