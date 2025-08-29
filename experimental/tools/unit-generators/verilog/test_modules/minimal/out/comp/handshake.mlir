module {
  handshake.func @minimal(%arg0: !handshake.channel<i32>, %arg1: !handshake.control<>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["x", "start"], resNames = ["out0", "end"]} {
    end {handshake.bb = 0 : ui32, handshake.name = "end0"} %arg0, %arg1 : <i32>, <>
  }
}

