module {
  handshake.func @sup_gamma_mux1_lhs(%cond: !handshake.channel<i1>, %dataT: !handshake.channel<i1>, %dataF: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["cond", "dataT", "dataF"], resNames = ["res"]} {
    %cond_forked:3 = fork [3] %cond {handshake.name = "cond_fork"} : <i1>
    %dataT_passed = passer %dataT [%cond_forked#0] {handshake.name = "dataT_passer"} : <i1>, <i1>
    %cond_not = not %cond_forked#1 {handshake.name = "not"} : <i1>
    %dataF_passed = passer %dataF [%cond_not] {handshake.name = "dataF_passer"} : <i1>, <i1>
    %res = mux %cond_forked#2 [%dataF_passed, %dataT_passed] {handshake.name = "res_mux"} : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.name = "end0"} %res : <i1>
  }
}
