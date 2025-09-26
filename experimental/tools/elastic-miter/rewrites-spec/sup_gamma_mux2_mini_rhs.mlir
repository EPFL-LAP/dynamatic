module {
  handshake.func @sup_gamma_mux2_mini_rhs(%cond: !handshake.channel<i1>, %dataT: !handshake.channel<i1>, %dataF: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["cond", "dataT", "dataF"], resNames = ["res"]} {
    %cond_forked:2 = fork [2] %cond {handshake.name = "cond_fork"} : <i1>

    %init = init %ri_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %src1 = source {handshake.name = "source1"} : <>
    %cst1 = constant %src1 {value = 1 : i1, handshake.name = "constant1"} : <>, <i1>
    %cond_buffered = buffer %cond_forked#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "ri_buff"} : <i1>
    %ri = mux %init [%cst1, %cond_buffered] {handshake.name = "ri"} : <i1>, [<i1>, <i1>] to <i1>
    %ri_forked:3 = fork [3] %ri {handshake.name = "ri_fork"} : <i1>
    %ri_buffered = buffer %ri_forked#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>

    %src2 = source {handshake.name = "source2"} : <>
    %cst2 = constant %src2 {value = 1 : i1, handshake.name = "constant2"} : <>, <i1>
    %cond_muxed = mux %ri_forked#1 [%cst2, %cond_forked#1] {handshake.name = "cond_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %data_muxed = mux %ri_forked#2 [%dataF, %dataT] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %res = passer %data_muxed [%cond_muxed] {handshake.name = "passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %res : <i1>
  }
}
