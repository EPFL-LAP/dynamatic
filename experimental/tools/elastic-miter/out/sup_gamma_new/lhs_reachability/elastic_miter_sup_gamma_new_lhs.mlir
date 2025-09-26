module {
  handshake.func @sup_gamma_new_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %0:5 = fork [5] %arg2 {handshake.name = "c1_fork"} : <i1>
    %1:3 = fork [3] %arg3 {handshake.name = "c2_fork"} : <i1>
    %2 = passer %1#0[%0#0] {handshake.name = "c21_1"} : <i1>, <i1>
    %3 = passer %1#1[%0#1] {handshake.name = "c21_2"} : <i1>, <i1>
    %4 = not %1#2 {handshake.name = "not"} : <i1>
    %5 = passer %4[%0#2] {handshake.name = "c2inv1"} : <i1>, <i1>
    %6 = passer %arg0[%0#3] {handshake.name = "a1_passer1"} : <i1>, <i1>
    %7 = passer %6[%2] {handshake.name = "a1_passer2"} : <i1>, <i1>
    %8 = passer %arg1[%0#4] {handshake.name = "a2_passer1"} : <i1>, <i1>
    %9 = passer %8[%5] {handshake.name = "a2_passer2"} : <i1>, <i1>
    %10 = mux %3 [%9, %7] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %10 : <i1>
  }
}
