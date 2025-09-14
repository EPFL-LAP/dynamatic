module {
  handshake.func @simpleInduction_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "C_in", "D_in"], resNames = ["B_out"]} {
    %0 = source {handshake.name = "source"} : <>
    %1 = constant %0 {handshake.name = "constant", value = false} : <>, <i1>
    %2 = mux %arg1 [%1, %arg2] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %3 = passer %arg0[%2] {handshake.name = "p1"} : <i1>, <i1>
    end {handshake.name = "end"} %3 : <i1>
  }
}
