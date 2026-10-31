module {
  handshake.func @h_lhs(%D: <i32>, %C: <i1>) -> (A: <i32>, B: <i32>) {
    %t, %f = cond_br %C, %D {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i32>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %a, %b = fork [2] %f {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %a, %b : <i32>, <i32>
  }
}