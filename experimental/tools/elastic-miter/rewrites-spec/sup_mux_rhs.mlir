module {
  handshake.func @j_rhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["In1", "In2", "Ctrl"], resNames = ["Out1"]} {
    %ctrl_init = init %ctrl_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_ctrl", hw.parameters = {INITIAL_TOKEN = 0 : i1, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %init_forked:2 = fork [2] %ctrl_init {handshake.bb = 1 : ui32, handshake.name = "fork_init"} : <i1>

    %data = mux %init_forked#0 [%b, %a] {handshake.bb = 1 : ui32, handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %src = source {handshake.bb = 1 : ui32, handshake.name = "source"}: <>
    %const_1 = constant %src {value = 1 : i1, handshake.bb = 1 : ui32, handshake.name = "const"} : <>, <i1>

    %ctrl_mux = mux %init_forked#1 [%const_1, %ctrl] {handshake.bb = 1 : ui32, handshake.name = "ctrl_mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %ctrl_buf = buffer %ctrl_mux {handshake.bb = 1 : ui32, handshake.name = "ctrl_buf", hw.parameters = {NUM_SLOTS = 1 : ui32, BUFFER_TYPE="ONE_SLOT_BREAK_DV", TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>

    %ctrl_forked:2 = fork [2] %ctrl_buf {handshake.bb = 1 : ui32, handshake.name = "fork_data_mux"} : <i1>
    %ctrl_not = not %ctrl_forked#1 {handshake.bb = 1 : ui32, handshake.name = "not_ctrl"} : <i1>

    %t, %f = cond_br %ctrl_not, %data {handshake.bb = 1 : ui32, handshake.name = "supp_br"} : <i1>, <i1>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink"} : <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %f : <i1>
  }
}
