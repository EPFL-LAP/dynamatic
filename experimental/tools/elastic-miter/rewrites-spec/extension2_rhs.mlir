module {
  handshake.func @extension2_rhs(%arg: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["arg"], resNames = ["res"]} {
    %arg_forked:2 = fork [2] %arg {handshake.name = "arg_fork"} : <i1>
    %arg_buffered = buffer %arg_forked#0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "arg_buff"} : <i1>

    %ri_buffered = buffer %ri_forked#0, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %inited = init %ri_buffered {handshake.name = "initop", initToken = 0 : ui1} : <i1>
    %src1 = source {handshake.name = "source1"} : <>
    %cst1 = constant %src1 {value = 1 : i1, handshake.name = "constant1"} : <>, <i1>
    %ried = mux %inited [%cst1, %arg_buffered] {handshake.name = "ri_mux"} : <i1>, [<i1>, <i1>] to <i1>
    %ri_forked:3 = fork [3] %ried {handshake.name = "ri_fork"} : <i1>

    %src2 = source {handshake.name = "source2"} : <>
    %cst2 = constant %src2 {value = 1 : i1, handshake.name = "constant2"} : <>, <i1>
    %muxed = mux %ri_forked#1 [%cst2, %arg_forked#1] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>

    %passed = passer %ri_forked#2 [%muxed] {handshake.name = "passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %passed : <i1>
  }
}
