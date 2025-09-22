module {
  handshake.func @general_sup_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, %arg4: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["sel", "ctrl1", "ctrl2", "a_in", "b_in"], resNames = ["res"]} {
    %0:2 = fork [2] %arg0 {handshake.name = "fork_sel"} : <i1>
    %1 = mux %0#0 [%arg1, %arg2] {handshake.name = "sel_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %2 = mux %0#1 [%arg3, %arg4] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %3 = passer %2[%1] {handshake.name = "data_passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %3 : <i1>
  }
}
