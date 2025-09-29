module {
  handshake.func @interpInduction_lhs(%short: !handshake.channel<i1>, %long: !handshake.channel<i1>) -> (!handshake.channel<i1>) attributes {argNames = ["short_in", "long_in"], resNames = ["B_out"]} {
    %long_forked:2 = fork [2] %long {handshake.name = "long_fork"} : <i1>
    %b = spec_v2_interpolator %short, %long_forked#0 {handshake.name = "interpolate"} : <i1>
    %c = spec_v2_repeating_init %long_forked#1 {handshake.name = "spec_v2_repeating_init", initToken = 1 : ui1} : <i1>
    %c_buf = buffer %c, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer", debugCounter = false} : <i1>
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 0 : i1, handshake.name = "constant"} : <>, <i1>
    %muxed = mux %c_buf [%cst, %b] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %muxed : <i1>
  }
}
