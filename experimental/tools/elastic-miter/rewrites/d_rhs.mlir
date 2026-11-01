module {
  handshake.func @d_rhs(%D: <i32>, %C: <i1>) -> (A: <i32>) {
    sink %C {handshake.bb = 1 : ui32, handshake.name = "sink"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %D : <i32>
  }
}