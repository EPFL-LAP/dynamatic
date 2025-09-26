module {
  handshake.func @sup_gamma_mux2_mini_lhs(%cond: !handshake.channel<i1>, %dataT: !handshake.channel<i1>, %dataF: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["cond", "dataT", "dataF"], resNames = ["res"]} {
    %cond_forked:2 = fork [2] %cond {handshake.name = "cond_fork"} : <i1>
    %dataT_passed = passer %dataT [%cond_forked#0] {handshake.name = "dataT_passer"} : <i1>, <i1>
    %res = mux %cond_forked#1 [%dataF, %dataT_passed] {handshake.name = "res_mux"} : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.name = "end0"} %res : <i1>
  }
}
