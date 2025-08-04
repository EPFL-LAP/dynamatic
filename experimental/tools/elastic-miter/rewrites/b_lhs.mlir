module {
  handshake.func @b_lhs(%x: !handshake.channel<i32>, %y: !handshake.channel<i32>, %d: !handshake.control<>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.control<>) attributes {argNames = ["Xd", "Yd","Dd", "Cd"], resNames = ["A", "B"]} {
    %a = mux %index [%x, %y] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i32>, <i32>] to <i32>
    %c_forked:2 = fork [2] %c {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>
    %c_not = not %c_forked#0 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %result, %index = control_merge [%d, %f_0_buf] {handshake.bb = 1 : ui32, handshake.name = "cmerge"} : [<>, <>] to <>, <i1>
    %result_forked:2 = fork [2] %result {handshake.bb = 1 : ui32, handshake.name = "fork_result"} : <>
    %t_1, %f_1 = cond_br %c_forked#1, %result_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <>
    %t_0, %f_0 = cond_br %c_not, %result_forked#0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <>
    %f_0_buf = buffer %f_0 {handshake.bb = 1 : ui32, handshake.name = "buf", hw.parameters = {NUM_SLOTS = 1 : ui32, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %a, %f_1 : <i32>, <>
  }
}