module {
  handshake.func @general_sup_mux_lhs(%sel: !handshake.channel<i1>, %ctrl1: !handshake.channel<i1>, %ctrl2: !handshake.channel<i1>, %a: !handshake.channel<i1>, %b: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["sel", "ctrl1", "ctrl2", "a_in", "b_in"], resNames = ["res"]} {
    %sel_forked:2 = fork [2] %sel {handshake.name = "fork_sel"} : <i1>
    %ctrl1_forked:2 = fork [2] %ctrl1 {handshake.name = "fork_ctrl1"} : <i1>
    %ctrl2_forked:2 = fork [2] %ctrl2 {handshake.name = "fork_ctrl2"} : <i1>
    %ctrl_muxed = mux %sel_forked#0 [%ctrl1_forked#0, %ctrl2_forked#0] {handshake.name = "sel_mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %sel_passed = passer %sel_forked#1 [%ctrl_muxed] {handshake.name = "sel_passer"} : <i1>, <i1>
    %a_passed = passer %a [%ctrl1_forked#1] {handshake.name = "a_passer"} : <i1>, <i1>
    %b_passed = passer %b [%ctrl2_forked#1] {handshake.name = "b_passer"} : <i1>, <i1>
    %data_muxed = mux %sel_passed [%a_passed, %b_passed] {handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %data_muxed : <i1>
  }
}
