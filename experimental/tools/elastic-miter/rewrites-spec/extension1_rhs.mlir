module {
  handshake.func @extension1_rhs(%arg: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["arg"], resNames = ["res"]} {
    %init = init %not_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %init_forked:3 = fork [3] %init {handshake.name = "cst_dup_fork"} : <i1>
    %not = not %init_forked#2 {handshake.name = "not"} : <i1>
    %not_buffered = buffer %not, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>


    %arg_forked:2 = fork [2] %arg {handshake.name = "arg_fork"} : <i1>
    %arg_not = not %arg_forked#0 {handshake.name = "arg_not"} : <i1>
    %arg_mux = mux %init_forked#0 [%arg_not, %arg_forked#1] {handshake.name = "arg_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %init_passed = passer %init_forked#1 [%arg_mux] {handshake.name = "cst_passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %init_passed : <i1>
  }
}
