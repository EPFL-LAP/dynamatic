module {
  handshake.func @sup_mux_rhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["In1", "In2", "Ctrl"], resNames = ["Out1"]} {
    %spec_ctrl = spec_v2_repeating_init %ctrl {handshake.bb = 1 : ui32, handshake.name = "ri"} : <i1>
    %ctrl_forked:2 = fork [2] %spec_ctrl {handshake.bb = 1 : ui32, handshake.name = "fork_data_mux"} : <i1>
    %ctrl_init = init %ctrl_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_ctrl", hw.parameters = {INITIAL_TOKEN = 0 : i1, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>

    %data = mux %ctrl_init [%b, %a] {handshake.bb = 1 : ui32, handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %passer_out = passer %data [%ctrl_forked#1] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %passer_out : <i1>
  }
}
