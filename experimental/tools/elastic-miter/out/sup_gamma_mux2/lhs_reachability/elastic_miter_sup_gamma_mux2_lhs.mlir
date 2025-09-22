module {
  handshake.func @sup_gamma_mux2_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["cond", "dataT", "dataF"], resNames = ["res"]} {
    %0:3 = fork [3] %arg0 {handshake.name = "cond_fork"} : <i1>
    %1 = passer %arg1[%0#0] {handshake.name = "dataT_passer"} : <i1>, <i1>
    %2 = not %0#1 {handshake.name = "not"} : <i1>
    %3 = passer %arg2[%2] {handshake.name = "dataF_passer"} : <i1>, <i1>
    %4 = mux %0#2 [%3, %1] {handshake.name = "res_mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %4 : <i1>
  }
}
