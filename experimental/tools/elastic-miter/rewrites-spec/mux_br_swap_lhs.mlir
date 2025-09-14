module {
  handshake.func @mux_br_swap_lhs(%c1: !handshake.channel<i1>, %c2: !handshake.channel<i1>, %a1: !handshake.channel<i1>, %a2: !handshake.channel<i1>) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["c1", "c2", "a1", "a2"], resNames = ["out1", "out2"]} {
    %mux_out = mux %c2 [%a2, %a1] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %trueRes, %falseRes = cond_br %c1, %mux_out {handshake.name = "cond_br"} : <i1>, <i1>
    end {handshake.name = "end0"} %trueRes, %falseRes : <i1>, <i1>
  }
}
