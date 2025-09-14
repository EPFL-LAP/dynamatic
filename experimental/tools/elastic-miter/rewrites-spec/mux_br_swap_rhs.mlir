module {
  handshake.func @mux_br_swap_rhs(%c1: !handshake.channel<i1>, %c2: !handshake.channel<i1>, %a1: !handshake.channel<i1>, %a2: !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["c1", "c2", "a1", "a2"], resNames = ["out1", "out2"]} {
    %c1_forked:2 = fork [2] %c1 {handshake.name = "fork1"} : <i1>
    %c2_forked:2 = fork [2] %c2 {handshake.name = "fork2"} : <i1>
    %true_c1, %false_c1 = cond_br %c2_forked#0, %c1_forked#0 {handshake.name = "cond_br0"} : <i1>, <i1>
    %trueRes1, %falseRes1 = cond_br %true_c1, %a1 {handshake.name = "cond_br1"} : <i1>, <i1>
    %trueRes2, %falseRes2 = cond_br %false_c1, %a2 {handshake.name = "cond_br2"} : <i1>, <i1>
    %true_c2, %false_c2 = cond_br %c1_forked#1, %c2_forked#1 {handshake.name = "cond_br3"} : <i1>, <i1>
    %mux_true = mux %true_c2 [%trueRes2, %trueRes1] {handshake.name = "mux1"} : <i1>, [<i1>, <i1>] to <i1>
    %mux_false = mux %false_c2 [%falseRes2, %falseRes1] {handshake.name = "mux2"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %mux_true, %mux_false : <i1>, <i1>
  }
}
