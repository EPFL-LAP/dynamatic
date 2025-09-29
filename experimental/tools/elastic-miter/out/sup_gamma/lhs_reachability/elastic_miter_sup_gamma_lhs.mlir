module {
  handshake.func @sup_gamma_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %0:3 = fork [3] %arg2 {handshake.name = "c1_fork"} : <i1>
    %1:3 = fork [3] %arg3 {handshake.name = "c2_fork"} : <i1>
    %2 = passer %0#0[%1#0] {handshake.name = "passer1"} : <i1>, <i1>
    %3 = not %1#1 {handshake.name = "not"} : <i1>
    %4 = passer %0#1[%3] {handshake.name = "passer2"} : <i1>, <i1>
    %5 = passer %1#2[%0#2] {handshake.name = "passer3"} : <i1>, <i1>
    %6 = passer %arg0[%2] {handshake.name = "passer_a1"} : <i1>, <i1>
    %7 = passer %arg1[%4] {handshake.name = "passer_a2"} : <i1>, <i1>
    %8 = mux %5 [%7, %6] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %8 : <i1>
  }
}
