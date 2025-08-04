module {
  handshake.func @f_lhs(%d: !handshake.channel<i32>, %m: !handshake.channel<i1>, %n: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>) attributes {argNames = ["D", "M", "N"], resNames = ["A"]} {
    %mux_0 = mux %m_init [%d, %loop_out_0] {handshake.bb = 1 : ui32, handshake.name = "mux_0"}  : <i1>, [<i32>, <i32>] to <i32>
    %mux0_forked:2 = fork [2] %mux_0 {handshake.bb = 1 : ui32, handshake.name = "fork_mux0"} : <i32>
    %t_0, %f_0 = cond_br %m_not, %mux0_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i32>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %loop_out_0 = buffer %f_0 {handshake.bb = 1 : ui32, handshake.name = "comb_buf_0", hw.parameters = {NUM_SLOTS = 1 : ui32, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i32>
    %mux_1 = mux %n_init [%mux0_forked#0, %loop_out_1] {handshake.bb = 1 : ui32, handshake.name = "mux_1"}  : <i1>, [<i32>, <i32>] to <i32>
    %mux1_forked:2 = fork [2] %mux_1 {handshake.bb = 1 : ui32, handshake.name = "fork_mux1"} : <i32>
    %t_1, %f_1 = cond_br %n_not, %mux1_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <i32>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i32>
    %loop_out_1 = buffer %f_1 {handshake.bb = 1 : ui32, handshake.name = "comb_buf_1", hw.parameters = {NUM_SLOTS = 1 : ui32, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i32>
    %m_forked:2 = fork [2] %m {handshake.bb = 1 : ui32, handshake.name = "fork_m"} : <i1>
    %m_init = init %m_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_m", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    %m_not = not %m_forked#1 {handshake.bb = 1 : ui32, handshake.name = "not_m"} : <i1>
    %n_forked:2 = fork [2] %n {handshake.bb = 1 : ui32, handshake.name = "fork_n"} : <i1>
    %n_init = init %n_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_n", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    %n_not = not %n_forked#1 {handshake.bb = 1 : ui32, handshake.name = "not_n"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %mux1_forked#0 : <i32>
  }
}