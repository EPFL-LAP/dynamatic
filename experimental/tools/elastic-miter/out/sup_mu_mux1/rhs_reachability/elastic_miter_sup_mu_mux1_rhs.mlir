module {
  handshake.func @sup_mu_mux1_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["cond", "dataT", "dataF", "ctrl"], resNames = ["res"]} {
    %0:3 = fork [3] %arg3 {handshake.name = "ctrl_fork"} : <i1>
    %1 = mux %4 [%0#0, %arg0] {handshake.name = "ctrl_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %2:3 = fork [3] %1 {handshake.name = "ctrl_muxed_fork"} : <i1>
    %3 = buffer %2#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %4 = init %3 {handshake.name = "ctrl_init", initToken = 0 : ui1} : <i1>
    %5 = source {handshake.name = "source2"} : <>
    %6 = constant %5 {handshake.name = "constant2", value = true} : <>, <i1>
    %7 = init %2#1 {handshake.name = "initop2", initToken = 0 : ui1} : <i1>
    %8 = not %0#2 {handshake.name = "ctrl_not"} : <i1>
    %9 = mux %7 [%8, %6] {handshake.name = "mux_passer_ctrl"} : <i1>, [<i1>, <i1>] to <i1>
    %10 = passer %2#2[%9] {handshake.name = "ctrl_passer"} : <i1>, <i1>
    %11 = init %10 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %12:2 = fork [2] %11 {handshake.name = "init_fork"} : <i1>
    %13 = source {handshake.name = "source"} : <>
    %14 = constant %13 {handshake.name = "constant", value = true} : <>, <i1>
    %15 = mux %12#0 [%0#1, %14] {handshake.name = "ctrl_mux2"} : <i1>, [<i1>, <i1>] to <i1>
    %16 = mux %12#1 [%arg2, %arg1] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %17 = passer %16[%15] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %17 : <i1>
  }
}
