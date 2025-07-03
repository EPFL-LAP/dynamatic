module {
  handshake.func @a_rhs(%d: !handshake.channel<i1>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["D_in", "C_in"], resNames = ["T_out", "F_out"]} {
    %d_forked:2 = fork [2] %d {handshake.bb = 1 : ui32, handshake.name = "fork_data"} : <i1>
    %c_forked:2 = fork [2] %c {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>
    %c_not = not %c_forked#0 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %t_0, %f_0 = cond_br %c_not, %d_forked#0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_T"} : <i1>, <i1>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i1>
    %t_1, %f_1 = cond_br %c_forked#1, %d_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_F"} : <i1>, <i1>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %f_0, %f_1 : <i1>, <i1>
  }
}
