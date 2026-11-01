module {
  handshake.func @h_rhs(%D: <i32>, %C: <i1>) -> (A: <i32>, B: <i32>) {
    %d_forked:2 = fork [2] %D {handshake.bb = 1 : ui32, handshake.name = "fork_data"} : <i32>
    %c_forked:2 = fork [2] %C {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>
    %t_0, %f_0 = cond_br %c_forked#0, %d_forked#0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i32>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %t_1, %f_1 = cond_br %c_forked#1, %d_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <i32>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %f_0, %f_1 : <i32>, <i32>
  }
}