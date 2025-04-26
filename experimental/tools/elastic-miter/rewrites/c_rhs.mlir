module {
  handshake.func @c_rhs(%d: !handshake.control<>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.control<>) attributes {argNames = ["D", "C"], resNames = ["A", "B"]} {
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %c, %d : <i1>, <>
  }
}