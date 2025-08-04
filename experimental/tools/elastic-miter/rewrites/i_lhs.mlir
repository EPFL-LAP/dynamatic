module {
  handshake.func @i_lhs(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["D", "C"], resNames = ["A", "B"]} {

    %data = mux %ctrl_init [%d, %loop_out] {handshake.bb = 1 : ui32, handshake.name = "data_mux"}  : <i1>, [<i32>, <i32>] to <i32>
    %data_forked:3 = fork [3] %data {handshake.bb = 1 : ui32, handshake.name = "fork_data_mux"} : <i32>
    %t, %f = cond_br %ctrl_not, %data_forked#2 {handshake.bb = 1 : ui32, handshake.name = "supp_br"} : <i1>, <i32>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink"} : <i32>
    %loop_out = buffer %f {handshake.bb = 1 : ui32, handshake.name = "comb_buf", hw.parameters = {NUM_SLOTS = 1 : ui32, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i32>

    %ctrl_forked:2 = fork [2] %c {handshake.bb = 1 : ui32, handshake.name = "fork_ctrl"} : <i1>
    %ctrl_init = init %ctrl_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_ctrl", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    %ctrl_not = not %ctrl_forked#1 {handshake.bb = 1 : ui32, handshake.name = "not_ctrl"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %data_forked#0, %data_forked#1 : <i32>, <i32>
  }
}