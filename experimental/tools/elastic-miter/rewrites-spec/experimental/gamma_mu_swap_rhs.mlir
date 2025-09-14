module {
  handshake.func @gamma_mu_swap_rhs(%c1: !handshake.channel<i1>, %c2: !handshake.channel<i1>, %a1: !handshake.channel<i1>, %a2: !handshake.channel<i1>, %a3: !handshake.channel<i1>) -> (!handshake.channel<i1>) attributes {argNames = ["c1", "c2", "a1", "a2", "a3"], resNames = ["out"]} {
    %c1_forked:2 = fork [2] %c1 {handshake.name = "fork1"} : <i1>
    %and = andi %c1_forked#1, %c2 {handshake.name = "and"} : <i1>
    %and_forked:2 = fork [2] %and {handshake.name = "fork3"} : <i1>
    %and_not = not %and_forked#0 {handshake.name = "not"} : <i1>
    %sel1 = passer %c1_forked#0 [%and_not] {handshake.name = "passer"} : <i1>, <i1>
    %mu1_sel = init %sel1 {handshake.name = "init1", initToken = 0 : ui1} : <i1>
    %mu1_out = mux %mu1_sel [%a3, %a2] {handshake.name = "mu1"} : <i1>, [<i1>, <i1>] to <i1>
    %mu2_sel = init %and_forked#1 {handshake.name = "init2", initToken = 0 : ui1} : <i1>
    %out = mux %mu2_sel [%mu1_out, %a1] {handshake.name = "mu2"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}
