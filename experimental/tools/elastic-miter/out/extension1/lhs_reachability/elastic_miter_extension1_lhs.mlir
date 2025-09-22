module {
  handshake.func @extension1_lhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg"], resNames = ["res"]} {
    %0 = not %arg0 {handshake.name = "not1"} : <i1>
    %1 = not %0 {handshake.name = "not2"} : <i1>
    end {handshake.name = "end0"} %1 : <i1>
  }
}
