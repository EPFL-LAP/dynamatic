module {
  handshake.func @i_rhs(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["D", "C"], resNames = ["A", "B"]} {
    %d_forked:2 = fork [2] %d {handshake.bb = 1 : ui32, handshake.name = "fork_data"} : <i32>
    %c_forked:2 = fork [2] %c {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>

    %data_0 = mux %ctrl_init_0 [%d_forked#0, %loop_out_0] {handshake.bb = 1 : ui32, handshake.name = "data_mux_0"}  : <i1>, [<i32>, <i32>] to <i32>
    %data_forked_0:2 = fork [2] %data_0 {handshake.bb = 1 : ui32, handshake.name = "fork_data_mux_0"} : <i32>
    %t_0, %f_0 = cond_br %ctrl_not_0, %data_forked_0#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i32>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %loop_out_0 = buffer %f_0 {handshake.bb = 1 : ui32, handshake.name = "comb_buf_0", hw.parameters = {NUM_SLOTS = 1 : ui32, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i32>

    %ctrl_forked_0:2 = fork [2] %c_forked#0 {handshake.bb = 1 : ui32, handshake.name = "fork_ctrl_0"} : <i1>
    %ctrl_init_0 = init %ctrl_forked_0#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_ctrl_0", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    %ctrl_not_0 = not %ctrl_forked_0#1 {handshake.bb = 1 : ui32, handshake.name = "not_ctrl_0"} : <i1>


    %data_1 = mux %ctrl_init_1 [%d_forked#1, %loop_out_1] {handshake.bb = 1 : ui32, handshake.name = "data_mux_1"}  : <i1>, [<i32>, <i32>] to <i32>
    %data_forked_1:2 = fork [2] %data_1 {handshake.bb = 1 : ui32, handshake.name = "fork_data_mux_1"} : <i32>
    %t_1, %f_1 = cond_br %ctrl_not_1, %data_forked_1#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <i32>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i32>
    %loop_out_1 = buffer %f_1 {handshake.bb = 1 : ui32, handshake.name = "comb_buf_1", hw.parameters = {NUM_SLOTS = 1 : ui32, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i32>

    %ctrl_forked_1:2 = fork [2] %c_forked#1 {handshake.bb = 1 : ui32, handshake.name = "fork_ctrl_1"} : <i1>
    %ctrl_init_1 = init %ctrl_forked_1#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_ctrl_1", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    %ctrl_not_1 = not %ctrl_forked_1#1 {handshake.bb = 1 : ui32, handshake.name = "not_ctrl_1"} : <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %data_forked_0#0, %data_forked_1#0 : <i32>, <i32>
  }
}