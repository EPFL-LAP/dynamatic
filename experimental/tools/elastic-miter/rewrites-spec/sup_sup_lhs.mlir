module {
  handshake.func @sup_sup_lhs(%a: !handshake.channel<i1>, %cond1: !handshake.channel<i1>, %cond2: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "Cond1", "Cond2"], resNames = ["B_out"]} {
    %cond2_forked:2 = fork [2] %cond2 {handshake.name = "fork_cond2"} : <i1>
    %cond1_passer = passer %cond1 [%cond2_forked#0] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %a_passer = passer %a [%cond2_forked#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    %b = passer %a_passer [%cond1_passer] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <i1>, <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %b : <i1>
  }
}
