module {
  handshake.func @sup_gamma_mux1_rhs(%cond: !handshake.channel<i1>, %dataT: !handshake.channel<i1>, %dataF: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["cond", "dataT", "dataF"], resNames = ["res"]} {
    %init = init %not_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %init_forked:3 = fork [3] %init {handshake.name = "cst_dup_fork"} : <i1>
    %not = not %init_forked#0 {handshake.name = "not"} : <i1>
    %not_buffered = buffer %not, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>

    %cond_forked:2 = fork [2] %cond {handshake.name = "cond_fork"} : <i1>
    %cond_not = not %cond_forked#0 {handshake.name = "cond_not"} : <i1>
    %cond_muxed = mux %init_forked#1 [%cond_not, %cond_forked#1] {handshake.name = "cond_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %data_muxed = mux %init_forked#2 [%dataF, %dataT] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %res = passer %data_muxed [%cond_muxed] {handshake.name = "passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %res : <i1>
  }
}
