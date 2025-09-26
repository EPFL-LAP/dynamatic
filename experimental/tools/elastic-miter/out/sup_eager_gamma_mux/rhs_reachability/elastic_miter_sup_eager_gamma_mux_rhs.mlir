module {
  handshake.func @sup_eager_gamma_mux_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in", "ctrl"], resNames = ["res"]} {
    %0 = init %3 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %1:3 = fork [3] %0 {handshake.name = "cst_dup_fork"} : <i1>
    %2 = not %1#0 {handshake.name = "not"} : <i1>
    %3 = buffer %2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %4:2 = fork [2] %arg2 {handshake.name = "ctrl_fork"} : <i1>
    %5 = mux %1#1 [%4#1, %4#0] {handshake.name = "ctrl_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %6 = mux %1#2 [%arg1, %arg0] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %7 = passer %6[%5] {handshake.name = "res_passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %7 : <i1>
  }
}
