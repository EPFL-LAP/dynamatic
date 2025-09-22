module {
  handshake.func @general_sup_mux_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, %arg4: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["sel", "ctrl1", "ctrl2", "a_in", "b_in"], resNames = ["res"]} {
    %0:2 = fork [2] %arg0 {handshake.name = "fork_sel"} : <i1>
    %1:2 = fork [2] %arg1 {handshake.name = "fork_ctrl1"} : <i1>
    %2:2 = fork [2] %arg2 {handshake.name = "fork_ctrl2"} : <i1>
    %3 = mux %0#0 [%1#0, %2#0] {handshake.name = "sel_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %4 = passer %0#1[%3] {handshake.name = "sel_passer"} : <i1>, <i1>
    %5 = passer %arg3[%1#1] {handshake.name = "a_passer"} : <i1>, <i1>
    %6 = passer %arg4[%2#1] {handshake.name = "b_passer"} : <i1>, <i1>
    %7 = mux %4 [%5, %6] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %7 : <i1>
  }
}
