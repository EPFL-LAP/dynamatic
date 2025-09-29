module {
  handshake.func @sup_mu_mux1_lhs(%cond: !handshake.channel<i1>, %dataT: !handshake.channel<i1>, %dataF: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["cond", "dataT", "dataF", "ctrl"], resNames = ["res"]} {
    %dataF_passed = passer %dataF [%ctrl] {handshake.name = "dataF_passer"} : <i1>, <i1>
    %inited = init %cond {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %res = mux %inited [%dataF_passed, %dataT] {handshake.name = "res_mux"} : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.name = "end0"} %res : <i1>
  }
}
