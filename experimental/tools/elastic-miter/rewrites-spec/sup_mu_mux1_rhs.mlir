module {
  handshake.func @sup_mu_mux1_rhs(%cond: !handshake.channel<i1>, %dataT: !handshake.channel<i1>, %dataF: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["cond", "dataT", "dataF", "ctrl"], resNames = ["res"]} {
    %ctrl_forked:3 = fork [3] %ctrl {handshake.name = "ctrl_fork"} : <i1>
    %ctrl_muxed = mux %ctrl_inited [%ctrl_forked#0, %cond] {handshake.name = "ctrl_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %ctrl_muxed_forked:3 = fork [3] %ctrl_muxed {handshake.name = "ctrl_muxed_fork"} : <i1>
    %ctrl_muxed_buffered = buffer %ctrl_muxed_forked#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %ctrl_inited = init %ctrl_muxed_buffered {handshake.name = "ctrl_init", initToken = 0 : ui1} : <i1>

    %src2 = source {handshake.name = "source2"} : <>
    %cst2 = constant %src2 {value = 1 : i1, handshake.name = "constant2"} : <>, <i1>
    %inited2 = init %ctrl_muxed_forked#1 {handshake.name = "initop2", initToken = 0 : ui1} : <i1>
    %ctrl_not = not %ctrl_forked#2 {handshake.name = "ctrl_not"} : <i1>
    %passer_ctrl = mux %inited2 [%ctrl_not, %cst2] {handshake.name = "mux_passer_ctrl"} : <i1>, [<i1>, <i1>] to <i1>
    %passed = passer %ctrl_muxed_forked#2 [%passer_ctrl] {handshake.name = "ctrl_passer"} : <i1>, <i1>

    %inited = init %passed {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %init_forked:2 = fork [2] %inited {handshake.name = "init_fork"} : <i1>

    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 1 : i1, handshake.name = "constant"} : <>, <i1>

    %ctrl_muxed2 = mux %init_forked#0 [%ctrl_forked#1, %cst] {handshake.name = "ctrl_mux2"} : <i1>, [<i1>, <i1>] to <i1>

    %data = mux %init_forked#1 [%dataF, %dataT] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>

    %res = passer %data [%ctrl_muxed2] {handshake.name = "passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %res : <i1>
  }
}
