module {
  handshake.func @b_lhs(%x: !handshake.channel<i1>, %y: !handshake.channel<i1>, %d: !handshake.control<>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.control<>) attributes {argNames = ["Xd_in", "Yd_in", "Dd_in", "Cd_in"], resNames = ["A_out", "B_out"]} {
    %a = mux %index [%x, %y] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %c_forked:2 = fork [2] %c {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>
    %c_not = not %c_forked#0 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %result, %index = control_merge [%d, %f_0_buf] {handshake.bb = 1 : ui32, handshake.name = "cmerge"} : [<>, <>] to <>, <i1>
    %result_forked:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork_result"} : <>
    %t_1, %f_1 = cond_br %c_forked#1, %result_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <>
    %t_0, %f_0 = cond_br %c_not, %result_forked#0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <>
    %f_0_buf = buffer %f_0 {handshake.bb = 1 : ui32, handshake.name = "buf", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %a, %f_1 : <i1>, <>
  }
}
