module {
  handshake.func @interpInduction_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["short_in", "long_in"], resNames = ["B_out"]} {
    %0 = spec_v2_repeating_init %arg1 {handshake.name = "spec_v2_repeating_init", initToken = 1 : ui1} : <i1>
    %1 = buffer %0, bufferType = FIFO_BREAK_NONE, numSlots = 1 {debugCounter = false, handshake.name = "buffer"} : <i1>
    %2 = spec_v2_interpolator %arg0, %1 {handshake.name = "interpolate"} : <i1>
    end {handshake.name = "end0"} %2 : <i1>
  }
}
