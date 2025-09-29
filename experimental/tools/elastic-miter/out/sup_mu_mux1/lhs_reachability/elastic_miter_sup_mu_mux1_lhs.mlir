module {
  handshake.func @sup_mu_mux1_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["cond", "dataT", "dataF", "ctrl"], resNames = ["res"]} {
    %0 = passer %arg2[%arg3] {handshake.name = "dataF_passer"} : <i1>, <i1>
    %1 = init %arg0 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %2 = mux %1 [%0, %arg1] {handshake.name = "res_mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %2 : <i1>
  }
}
