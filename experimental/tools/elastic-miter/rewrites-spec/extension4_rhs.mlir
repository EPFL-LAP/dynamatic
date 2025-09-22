module {
  handshake.func @extension4_rhs(%arg: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["arg", "ctrl"], resNames = ["res"]} {
    %arg_forked:2 = fork [2] %arg {handshake.name = "arg_fork"} : <i1>
    %ctrl_forked:2 = fork [2] %ctrl {handshake.name = "ctrl_fork"} : <i1>
    %arg_ctrl_and = andi %arg_forked#0, %ctrl_forked#0 {handshake.name = "andi"} : <i1>
    %arg_ctrl_and_buffered = buffer %arg_ctrl_and, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "and_buff"} : <i1>
    %ctrl_not = not %ctrl_forked#1 {handshake.name = "ctrl_not"} : <i1>
    %arg_ctrl_not_or = ori %arg_forked#1, %ctrl_not {handshake.name = "ori"} : <i1>

    %ri_buffered = buffer %ried, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %ri_inited = init %ri_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %ri_forked:3 = fork [3] %ri_inited {handshake.name = "ri_fork"} : <i1>

    %src1 = source {handshake.name = "source1"} : <>
    %cst1 = constant %src1 {value = 1 : i1, handshake.name = "constant1"} : <>, <i1>
    %ried = mux %ri_forked#0 [%cst1, %arg_ctrl_not_or] {handshake.name = "ri_mux"} : <i1>, [<i1>, <i1>] to <i1>

    %src2 = source {handshake.name = "source2"} : <>
    %cst2 = constant %src2 {value = 1 : i1, handshake.name = "constant2"} : <>, <i1>
    %muxed = mux %ri_forked#1 [%cst2, %arg_ctrl_and_buffered] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>

    %passed = passer %ri_forked#2 [%muxed] {handshake.name = "passer"} : <i1>, <i1>
    %passed_buffered = buffer %passed, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "p_buff"} : <i1>

    end {handshake.name = "end0"} %passed_buffered : <i1>
  }
}
