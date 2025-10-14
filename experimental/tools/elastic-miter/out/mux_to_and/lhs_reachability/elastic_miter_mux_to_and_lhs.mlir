module {
  handshake.func @mux_to_and_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in"], resNames = ["C_out"]} {
    %0 = source {handshake.name = "source"} : <>
    %1 = constant %0 {handshake.name = "constant", value = false} : <>, <i1>
    %2:2 = fork [2] %arg0 {handshake.name = "a_fork"} : <i1>
    %3 = passer %arg1[%2#0] {handshake.name = "passer1"} : <i1>, <i1>
    %4 = mux %2#1 [%1, %3] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %4 : <i1>
  }
}
