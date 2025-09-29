module {
  handshake.func @repeating_init_rhs(%in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 1 : i1, handshake.name = "constant"} : <>, <i1>
    %ri = mux %sel_buffered [%cst, %in] {handshake.name = "mux"} : <i1> , [<i1>, <i1>] to <i1>
    %ri_forked:2 = fork [2] %ri {handshake.name = "ri_fork"} : <i1>
    %sel = init %ri_forked#0 {handshake.name = "newInit", initToken = 0 : ui1} : <i1>
    %sel_buffered = buffer %sel, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %ri_forked#1 : <i1>
  }
}
