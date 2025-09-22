module {
  handshake.func @general_sup_mux_rhs(%sel: !handshake.channel<i1>, %ctrl1: !handshake.channel<i1>, %ctrl2: !handshake.channel<i1>, %a: !handshake.channel<i1>, %b: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["sel", "ctrl1", "ctrl2", "a_in", "b_in"], resNames = ["res"]} {
    %sel_forked:2 = fork [2] %sel {handshake.name = "fork_sel"} : <i1>
    %ctrl_muxed = mux %sel_forked#0 [%ctrl1, %ctrl2] {handshake.name = "sel_mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %data_muxed = mux %sel_forked#1 [%a, %b] {handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %data_passed = passer %data_muxed [%ctrl_muxed] {handshake.name = "data_passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %data_passed : <i1>
  }
}
