module {
  handshake.func @sup_gamma_mux1_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["cond", "dataT", "dataF"], resNames = ["res"]} {
    %0 = init %3 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %1:3 = fork [3] %0 {handshake.name = "cst_dup_fork"} : <i1>
    %2 = not %1#0 {handshake.name = "not"} : <i1>
    %3 = buffer %2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %4:2 = fork [2] %arg0 {handshake.name = "cond_fork"} : <i1>
    %5 = not %4#0 {handshake.name = "cond_not"} : <i1>
    %6 = mux %1#1 [%5, %4#1] {handshake.name = "cond_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %7 = mux %1#2 [%arg2, %arg1] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %8 = passer %7[%6] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %8 : <i1>
  }
}
