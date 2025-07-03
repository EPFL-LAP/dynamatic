module {
  handshake.func @f_rhs(%d: !handshake.channel<i1>, %m: !handshake.channel<i1>, %n: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["D_in", "M_in", "N_in"], resNames = ["A_out"]} {
    %src = source {handshake.bb = 1 : ui32, handshake.name = "source"}: <>
    %const_1 = constant %src {value = 1 : i1, handshake.bb = 1 : ui32, handshake.name = "const"} : <>, <i1>
    %ctrl = mux %n [%m, %const_1] {handshake.bb = 1 : ui32, handshake.name = "ctrl_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %data = mux %ctrl_init [%d, %loop_out] {handshake.bb = 1 : ui32, handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %data_forked:2 = fork [2] %data {handshake.bb = 1 : ui32, handshake.name = "fork_data_mux"} : <i1>
    %t, %f = cond_br %ctrl_not, %data_forked#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br"} : <i1>, <i1>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink"} : <i1>
    %loop_out = buffer %f {handshake.bb = 1 : ui32, handshake.name = "comb_buf", hw.parameters = {BUFFER_TYPE = "ONE_SLOT_BREAK_DV", NUM_SLOTS = 1 : ui32, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>

    %ctrl_forked:2 = fork [2] %ctrl {handshake.bb = 1 : ui32, handshake.name = "fork_ctrl"} : <i1>
    %ctrl_init = init %ctrl_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_ctrl", hw.parameters = {INITIAL_TOKEN = 0 : i1, BUFFER_TYPE = "ONE_SLOT_BREAK_DV"}} : <i1>
    %ctrl_not = not %ctrl_forked#1 {handshake.bb = 1 : ui32, handshake.name = "not_ctrl"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %data_forked#0 : <i1>
  }
}
