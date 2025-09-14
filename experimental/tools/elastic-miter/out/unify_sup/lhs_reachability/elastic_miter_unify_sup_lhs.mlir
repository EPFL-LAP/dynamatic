module {
  handshake.func @unify_sup_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "Cond1", "Cond2"], resNames = ["B_out"]} {
    %0:2 = fork [2] %arg2 {handshake.name = "fork_cond2"} : <i1>
    %1 = passer %arg1[%0#0] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %2 = passer %arg0[%0#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    %3 = passer %2[%1] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %3 : <i1>
  }
}
