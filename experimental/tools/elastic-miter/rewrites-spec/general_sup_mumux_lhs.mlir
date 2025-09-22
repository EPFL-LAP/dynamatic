module {
  handshake.func @general_sup_mumux_lhs(%dataT: !handshake.channel<i1>, %dataF: !handshake.channel<i1>, %cond: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["dataT", "dataF", "cond", "ctrl"], resNames = ["ctrlRes", "dataRes"]} {
    %ctrl_forked:2 = fork [2] %ctrl {handshake.name = "ctrl_fork"} : <i1>
    %cond_passed = passer %cond [%ctrl_forked#0] {handshake.name = "cond_passer"} : <i1>, <i1>
    %dataT_passed = passer %dataT [%ctrl_forked#1] {handshake.name = "dataT_passer"} : <i1>, <i1>

    %ri_forked:2 = fork [2] %ri {handshake.name = "ri_fork"} : <i1>
    %ri_buffered = buffer %ri_forked#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "ri_buff"} : <i1>
    %inited = init %ri_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %init_forked:2 = fork [2] %inited {handshake.name = "init_fork"} : <i1>

    %src1 = source {handshake.name = "source1"} : <>
    %cst1 = constant %src1 {value = 1 : i1, handshake.name = "constant1"} : <>, <i1>
    %ri = mux %init_forked#0 [%cst1, %cond_passed] {handshake.name = "ri"} : <i1>, [<i1>, <i1>] to <i1>

    %dataRes = mux %init_forked#1 [%dataF, %dataT_passed] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.name = "end0"} %ri_forked#1, %dataRes : <i1>, <i1>
  }
}
