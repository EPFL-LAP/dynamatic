module {
  handshake.func @context(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["ctrl"], resNames = ["ctrl"]} {
    end {handshake.bb = 1 : ui32, handshake.name = "end"} %arg0 : <i1>
  }
}
