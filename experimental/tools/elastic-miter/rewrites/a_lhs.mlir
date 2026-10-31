module {
  handshake.func @a_lhs(%D: <i32>, %C: <i1>) -> (T: <i32>, F: <i32>) {
    %t, %f = cond_br %C, %D {handshake.bb = 1 : ui32, handshake.name = "branch"} : <i1>, <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %t, %f : <i32>, <i32>
  }
}