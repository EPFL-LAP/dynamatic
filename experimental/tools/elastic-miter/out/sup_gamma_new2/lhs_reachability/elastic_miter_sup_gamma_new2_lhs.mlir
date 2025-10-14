module {
  handshake.func @sup_gamma_new2_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %0:3 = fork [3] %arg2 {handshake.name = "c1_fork"} : <i1>
    %1 = passer %arg3[%0#0] {handshake.name = "c2_passer"} : <i1>, <i1>
    %2:3 = fork [3] %1 {handshake.name = "c2_fork"} : <i1>
    %3 = not %2#2 {handshake.name = "not"} : <i1>
    %4 = passer %arg0[%0#1] {handshake.name = "a1_passer1"} : <i1>, <i1>
    %5 = passer %4[%2#0] {handshake.name = "a1_passer2"} : <i1>, <i1>
    %6 = passer %arg1[%0#2] {handshake.name = "a2_passer1"} : <i1>, <i1>
    %7 = passer %6[%3] {handshake.name = "a2_passer2"} : <i1>, <i1>
    %8 = mux %2#1 [%7, %5] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %8 : <i1>
  }
}
