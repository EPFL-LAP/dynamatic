module {
  handshake.func @interpInduction_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["short_in", "long_in"], resNames = ["B_out"]} {
    %0:2 = fork [2] %arg1 {handshake.name = "long_fork"} : <i1>
    %1 = spec_v2_interpolator %arg0, %0#0 {handshake.name = "interpolate"} : <i1>
    %2 = spec_v2_repeating_init %0#1 {handshake.name = "spec_v2_repeating_init", initToken = 1 : ui1} : <i1>
    %3 = buffer %2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "buffer"} : <i1>
    %4 = source {handshake.name = "source"} : <>
    %5 = constant %4 {handshake.name = "constant", value = false} : <>, <i1>
    %6 = mux %3 [%5, %1] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %6 : <i1>
  }
}
