module {
  handshake.func @extension3_rhs(%arg: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["arg"], resNames = ["res"]} {
    %arg_nd = ndwire %arg {handshake.name = "arg_ndwire"} : <i1>
    %arg_forked:2 = fork [2] %arg_nd {handshake.name = "arg_fork"} : <i1>

    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 1 : i1, handshake.name = "constant"} : <>, <i1>
    %arg_nd2 = ndwire %arg_forked#0 {handshake.name = "arg_ndwire2"} : <i1>
    %cst_merged = merge %arg_nd2, %cst {handshake.name = "cst_merge"} : <i1>
    %cst_buffered = buffer %cst_merged, bufferType = ONE_SLOT_BREAK_R, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>

    %cst_forked:2 = fork [2] %cst_buffered {handshake.name = "cst_fork"} : <i1>

    %src2 = source {handshake.name = "source2"} : <>
    %cst2 = constant %src2 {value = 1 : i1, handshake.name = "constant2"} : <>, <i1>
    %sel = mux %cst_forked#0 [%cst2, %arg_forked#1] {handshake.name = "ri_mux"} : <i1>, [<i1>, <i1>] to <i1>

    %passed = passer %cst_forked#1 [%sel] {handshake.name = "passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %passed : <i1>
  }
}
