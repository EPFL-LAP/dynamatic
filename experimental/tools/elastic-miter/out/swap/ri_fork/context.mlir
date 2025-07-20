module {
  handshake.func @context(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in"], resNames = ["A_in"]} {
    end {handshake.bb = 1 : ui32, handshake.name = "end"} %arg0 : <i1>
  }
}
