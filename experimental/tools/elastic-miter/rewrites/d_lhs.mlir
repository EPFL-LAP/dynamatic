module {
  handshake.func @d_lhs(%d: !handshake.channel<i1>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["D_in", "C_in"], resNames = ["A_out"]} {
    %a = mux %index [%d, %f_0_buf2] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %c_forked:3 = fork [3] %c {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>
    %index = init %c_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    %c_not = not %c_forked#2 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %mux_forked:2 = fork [2] %a {handshake.bb = 1 : ui32, handshake.name = "fork_mux"} : <i1>
    %t_1, %f_1 = cond_br %c_forked#1, %mux_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i1>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i1>
    %t_0, %f_0 = cond_br %c_not, %mux_forked#0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <i1>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i1>
    %f_0_buf = buffer %f_0 {handshake.bb = 1 : ui32, handshake.name = "buf", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %f_0_buf2 = buffer %f_0_buf {handshake.bb = 1 : ui32, handshake.name = "buf2", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_R", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 0, V: 0, R: 1}>}} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %f_1 : <i1>
  }
}
