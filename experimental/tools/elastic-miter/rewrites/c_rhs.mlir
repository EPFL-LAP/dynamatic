module {
  handshake.func @c_rhs(%D: <>, %C: <i1>) -> (A: <i1>, B: <>) {
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %C, %D : <i1>, <>
  }
}