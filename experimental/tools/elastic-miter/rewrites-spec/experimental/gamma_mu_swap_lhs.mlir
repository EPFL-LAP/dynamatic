module {
  handshake.func @gamma_mu_swap_lhs(%c1: !handshake.channel<i1>, %c2: !handshake.channel<i1>, %a1: !handshake.channel<i1>, %a2: !handshake.channel<i1>, %a3: !handshake.channel<i1>) -> (!handshake.channel<i1>) attributes {argNames = ["c1", "c2", "a1", "a2", "a3"], resNames = ["out"]} {
    %c1_forked:2 = fork [2] %c1 {handshake.name = "fork"} : <i1>
    %c2_passer = passer %c2 [%c1_forked#0] {handshake.name = "passer"} : <i1>, <i1>
    %gamma_out = mux %c2_passer [%a2, %a1] {handshake.name = "gamma"} : <i1>, [<i1>, <i1>] to <i1>
    %mu_sel = init %c1_forked#1 {handshake.name = "init0", initToken = 0 : ui1} : <i1>
    %mu_out = mux %mu_sel [%a3, %gamma_out] {handshake.name = "mu"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %mu_out : <i1>
  }
}
