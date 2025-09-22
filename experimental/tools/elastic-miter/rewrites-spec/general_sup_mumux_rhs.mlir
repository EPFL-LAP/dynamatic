module {
  handshake.func @general_sup_mumux_rhs(%dataT: !handshake.channel<i1>, %dataF: !handshake.channel<i1>, %cond: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["dataT", "dataF", "cond", "ctrl"], resNames = ["ctrlRes", "dataRes"]} {
    %ctrl_forked:2 = fork [2] %ctrl {handshake.name = "ctrl_fork"} : <i1>
    %ctrl_not = not %ctrl_forked#0 {handshake.name = "ctrl_not"} : <i1>
    %ored = ori %ctrl_not, %cond {handshake.name = "ori"} : <i1>

    %ri_forked:2 = fork [2] %ri {handshake.name = "ri_fork"} : <i1>
    %ri_buffered = buffer %ri_forked#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "ri_buff"} : <i1>
    %inited = init %ri_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %init_forked:3 = fork [3] %inited {handshake.name = "init_fork"} : <i1>

    %src1 = source {handshake.name = "source1"} : <>
    %cst1 = constant %src1 {value = 1 : i1, handshake.name = "constant1"} : <>, <i1>
    %ri = mux %init_forked#0 [%cst1, %ored] {handshake.name = "ri"} : <i1>, [<i1>, <i1>] to <i1>

    %dataRes = mux %init_forked#1 [%dataF, %dataT] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>

    %src2 = source {handshake.name = "source2"} : <>
    %cst2 = constant %src2 {value = 1 : i1, handshake.name = "constant2"} : <>, <i1>
    %ctrl_muxed = mux %init_forked#2 [%cst2, %ctrl_forked#1] {handshake.name = "ctrl_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %ctrl_muxed_forked:2 = fork [2] %ctrl_muxed {handshake.name = "ctrl_muxed_fork"} : <i1>

    %ctrlRes_passed = passer %ri_forked#1 [%ctrl_muxed_forked#0] {handshake.name = "ctrlRes_passer"} : <i1>, <i1>
    %dataRes_passed = passer %dataRes [%ctrl_muxed_forked#1] {handshake.name = "dataRes_passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %ctrlRes_passed, %dataRes_passed : <i1>, <i1>
  }
}
