module {
  handshake.func @d_rhs(%d: !handshake.channel<i1>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["D_in", "C_in"], resNames = ["A_out"]} {
    sink %c {handshake.bb = 1 : ui32, handshake.name = "sink"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %d : <i1>
  }
}
