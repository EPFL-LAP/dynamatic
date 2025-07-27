module {
  handshake.func @context(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["a1", "a2", "c1", "c2"]} {
    end {handshake.bb = 1 : ui32, handshake.name = "end"} %arg0, %arg1, %arg2, %arg3 : <i1>, <i1>, <i1>, <i1>
  }
}
