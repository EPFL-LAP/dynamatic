module {
  handshake.func @g_lhs(%D: <i32>, %M: <i1>, %N: <i1>) -> (A: <i32>) {
    %t_0, %f_0 = cond_br %M, %D {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i32>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %t_1, %f_1 = cond_br %N, %f_0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <i32>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %f_1 : <i32>
  }
}