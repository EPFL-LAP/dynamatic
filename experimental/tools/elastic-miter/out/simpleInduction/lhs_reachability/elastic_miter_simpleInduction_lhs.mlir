module {
  handshake.func @simpleInduction_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "C_in", "D_in"], resNames = ["B_out"]} {
    %0 = passer %arg0[%arg1] {handshake.name = "p1"} : <i1>, <i1>
    %1 = passer %0[%arg2] {handshake.name = "p2"} : <i1>, <i1>
    end {handshake.name = "end"} %1 : <i1>
  }
}
