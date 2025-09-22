module {
  handshake.func @extension2_lhs(%arg: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["arg"], resNames = ["res"]} {
    %arg1 = not %arg {handshake.name = "not1"} : <i1>
    %arg2 = not %arg1 {handshake.name = "not2"} : <i1>
    end {handshake.name = "end0"} %arg2 : <i1>
  }
}
