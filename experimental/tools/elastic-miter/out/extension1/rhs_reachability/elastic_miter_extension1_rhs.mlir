module {
  handshake.func @extension1_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["arg"], resNames = ["res"]} {
    %0 = init %3 {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %1:3 = fork [3] %0 {handshake.name = "cst_dup_fork"} : <i1>
    %2 = not %1#2 {handshake.name = "not"} : <i1>
    %3 = buffer %2, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %4:2 = fork [2] %arg0 {handshake.name = "arg_fork"} : <i1>
    %5 = not %4#0 {handshake.name = "arg_not"} : <i1>
    %6 = mux %1#0 [%5, %4#1] {handshake.name = "arg_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %7 = passer %1#1[%6] {handshake.name = "cst_passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %7 : <i1>
  }
}
