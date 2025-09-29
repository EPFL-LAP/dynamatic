module {
  handshake.func @repeating_init_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %0 = source {handshake.name = "source"} : <>
    %1 = constant %0 {handshake.name = "constant", value = true} : <>, <i1>
    %2 = mux %5 [%1, %arg0] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %3:2 = fork [2] %2 {handshake.name = "ri_fork"} : <i1>
    %4 = init %3#0 {handshake.name = "newInit", initToken = 0 : ui1} : <i1>
    %5 = buffer %4, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %3#1 : <i1>
  }
}
