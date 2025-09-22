module {
  handshake.func @extension3_lhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg"], resNames = ["res"]} {
    %0 = init %arg0 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    end {handshake.name = "end0"} %0 : <i1>
  }
}
