module {
  handshake.func @general_sup_mumux_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["dataT", "dataF", "cond", "ctrl"], resNames = ["ctrlRes", "dataRes"]} {
    %0:2 = fork [2] %arg3 {handshake.name = "ctrl_fork"} : <i1>
    %1 = passer %arg2[%0#0] {handshake.name = "cond_passer"} : <i1>, <i1>
    %2 = passer %arg0[%0#1] {handshake.name = "dataT_passer"} : <i1>, <i1>
    %3:2 = fork [2] %9 {handshake.name = "ri_fork"} : <i1>
    %4 = buffer %3#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "ri_buff"} : <i1>
    %5 = init %4 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %6:2 = fork [2] %5 {handshake.name = "init_fork"} : <i1>
    %7 = source {handshake.name = "source1"} : <>
    %8 = constant %7 {handshake.name = "constant1", value = true} : <>, <i1>
    %9 = mux %6#0 [%8, %1] {handshake.name = "ri"} : <i1>, [<i1>, <i1>] to <i1>
    %10 = mux %6#1 [%arg1, %2] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %3#1, %10 : <i1>, <i1>
  }
}
