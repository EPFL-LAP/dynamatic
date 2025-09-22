module {
  handshake.func @sup_gamma_mux2_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["cond", "dataT", "dataF"], resNames = ["res"]} {
    %0:3 = fork [3] %arg0 {handshake.name = "cond_fork"} : <i1>
    %1 = init %7 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %2 = source {handshake.name = "source1"} : <>
    %3 = constant %2 {handshake.name = "constant1", value = true} : <>, <i1>
    %4 = buffer %0#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "ri_buff"} : <i1>
    %5 = mux %1 [%3, %4] {handshake.name = "ri"} : <i1>, [<i1>, <i1>] to <i1>
    %6:3 = fork [3] %5 {handshake.name = "ri_fork"} : <i1>
    %7 = buffer %6#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %8 = source {handshake.name = "source2"} : <>
    %9 = constant %8 {handshake.name = "constant2", value = true} : <>, <i1>
    %10 = mux %6#1 [%9, %0#1] {handshake.name = "cond_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %11 = not %0#2 {handshake.name = "not"} : <i1>
    %12 = passer %arg2[%11] {handshake.name = "dataF_passer"} : <i1>, <i1>
    %13 = mux %6#2 [%12, %arg1] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %14 = passer %13[%10] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %14 : <i1>
  }
}
