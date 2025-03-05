module {
  handshake.func @c_lhs(%d: !handshake.control<>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.control<>) attributes {argNames = ["D", "C"], resNames = ["A", "B"]} {
    %d_forked:2 = fork [2] %d {handshake.bb = 1 : ui32, handshake.name = "fork_data"} : <>
    %c_forked:2 = fork [2] %c {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>
    %c_not = not %c_forked#0 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %t_0, %f_0 = cond_br %c_not, %d_forked#0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "sink_0"} : <>
    %t_1, %f_1 = cond_br %c_forked#1, %d_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "sink_1"} : <>
    %b, %a = control_merge [%f_0, %f_1] {handshake.bb = 1 : ui32, handshake.name = "cmerge"} : [<>, <>] to <>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %a, %b : <i1>, <>
  }
}