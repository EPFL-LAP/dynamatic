module {
  handshake.func @d_lhs(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>) attributes {argNames = ["Dd", "Cd"], resNames = ["A"]} {
    %a = mux %index [%d, %f_0_buf] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i32>, <i32>] to <i32>
    %c_forked:3 = fork [3] %c {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>
    %index = init %c_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    %c_not = not %c_forked#2 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %mux_forked:2 = fork [2] %a {handshake.bb = 1 : ui32, handshake.name = "fork_mux"} : <i32>
    %t_1, %f_1 = cond_br %c_forked#1, %mux_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i32>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %t_0, %f_0 = cond_br %c_not, %mux_forked#0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <i32>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i32>
    %f_0_buf = buffer %f_0 {handshake.bb = 1 : ui32, handshake.name = "buf", hw.parameters = {NUM_SLOTS = 1 : ui32, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %f_1 : <i32>
  }
}