module {
  handshake.func @sup_gamma_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %0 = mux %arg3 [%arg1, %arg0] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %1 = passer %0[%arg2] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %1 : <i1>
  }
}
