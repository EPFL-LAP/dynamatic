module {
  handshake.func @sup_gamma_new2_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %0:3 = fork [3] %arg3 {handshake.name = "c2_fork"} : <i1>
    %1 = passer %arg0[%0#0] {handshake.name = "a1_passer"} : <i1>, <i1>
    %2 = not %0#1 {handshake.name = "not"} : <i1>
    %3 = passer %arg1[%2] {handshake.name = "a2_passer"} : <i1>, <i1>
    %4 = mux %0#2 [%3, %1] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %5 = passer %4[%arg2] {handshake.name = "b_passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %5 : <i1>
  }
}
