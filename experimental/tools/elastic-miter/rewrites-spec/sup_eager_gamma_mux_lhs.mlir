module {
  handshake.func @sup_eager_gamma_mux_lhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "B_in", "ctrl"], resNames = ["res"]} {
    %init = init %not_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %init_forked:2 = fork [2] %init {handshake.name = "cst_dup_fork"} : <i1>
    %not = not %init_forked#0 {handshake.name = "not"} : <i1>
    %not_buffered = buffer %not, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>

    %ctrl_forked:2 = fork [2] %ctrl {handshake.name = "ctrl_fork"} : <i1>
    %a_passed = passer %a [%ctrl_forked#0] {handshake.name = "a_passer"} : <i1>, <i1>
    %b_passed = passer %b [%ctrl_forked#1] {handshake.name = "b_passer"} : <i1>, <i1>

    %res = mux %init_forked#1 [%b_passed, %a_passed] {handshake.name = "mux"}  : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.name = "end0"} %res : <i1>
  }
}
