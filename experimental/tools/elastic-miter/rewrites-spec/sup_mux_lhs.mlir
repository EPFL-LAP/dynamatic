module {
  handshake.func @sup_mux_lhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["In1", "In2", "Ctrl"], resNames = ["Out1"]} {

    %ctrl_forked:2 = fork [2] %ctrl {handshake.bb = 1 : ui32, handshake.name = "fork_data_mux"} : <i1>
    %ctrl_init = init %ctrl_forked#0 {handshake.bb = 1 : ui32, handshake.name = "init_buffer_ctrl", hw.parameters = {INITIAL_TOKEN = 0 : i1, TIMING = #handshake<timing {D: 1, V: 1, R: 0}>}} : <i1>
    %ctrl_not = not %ctrl_forked#1 {handshake.bb = 1 : ui32, handshake.name = "not_ctrl"} : <i1>

    %t, %f = cond_br %ctrl_not, %a {handshake.bb = 1 : ui32, handshake.name = "supp_br"} : <i1>, <i1>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink"} : <i1>

    %data = mux %ctrl_init [%b, %f] {handshake.bb = 1 : ui32, handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %data : <i1>
  }
}
