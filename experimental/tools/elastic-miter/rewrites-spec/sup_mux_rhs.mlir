module {
  handshake.func @sup_mux_rhs(%loopLiveIn: !handshake.channel<i1>, %iterLiveOut: !handshake.channel<i1>, %oldContinue: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["iterLiveIn"]} {
    %newContinue = spec_v2_repeating_init %oldContinue {handshake.name = "ri1", initToken = 1 : ui1} : <i1>
    %newContinue_buffered = buffer %newContinue, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer1", debugCounter = false} : <i1>
    %newSelector = init %newContinue_buffered {handshake.name = "newInit", initToken = 0 : ui1} : <i1>
    %data = mux %newSelector [%loopLiveIn, %iterLiveOut] {handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %newContinue2 = spec_v2_repeating_init %oldContinue {handshake.name = "ri2", initToken = 1 : ui1} : <i1>
    %newContinue_buffered2 = buffer %newContinue2, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer2", debugCounter = false} : <i1>
    %passerOut = passer %data [%newContinue_buffered2] {handshake.name = "passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %passerOut : <i1>
  }
}
