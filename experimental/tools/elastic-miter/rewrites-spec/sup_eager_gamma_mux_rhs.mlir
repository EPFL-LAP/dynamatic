module {
  handshake.func @sup_eager_gamma_mux_rhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "B_in", "ctrl"], resNames = ["res"]} {
    %init = init %not_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %init_forked:3 = fork [3] %init {handshake.name = "cst_dup_fork"} : <i1>
    %not = not %init_forked#0 {handshake.name = "not"} : <i1>
    %not_buffered = buffer %not, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>

    %ctrl_forked:2 = fork [2] %ctrl {handshake.name = "ctrl_fork"} : <i1>
    %ctrl_muxed = mux %init_forked#1 [%ctrl_forked#1, %ctrl_forked#0] {handshake.name = "ctrl_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %res = mux %init_forked#2 [%b, %a] {handshake.name = "mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %res_passed = passer %res [%ctrl_muxed] {handshake.name = "res_passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %res_passed : <i1>
  }
}
