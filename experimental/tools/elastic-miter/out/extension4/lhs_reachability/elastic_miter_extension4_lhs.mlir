module {
  handshake.func @extension4_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg", "ctrl"], resNames = ["res"]} {
    %0 = passer %arg0[%arg1] {handshake.name = "passer"} : <i1>, <i1>
    %1 = init %0 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    end {handshake.name = "end0"} %1 : <i1>
  }
}
