module {
  handshake.func @sup_mux_rhs(%loopLiveIn: !handshake.channel<i1>, %iterLiveOut: !handshake.channel<i1>, %oldContinue: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["iterLiveIn"]} {
    %oldContinue_buffered = buffer %oldContinue, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer1"} : <i1>
    %newContinue = spec_v2_repeating_init %oldContinue_buffered {handshake.name = "ri1", initToken = 1 : ui1} : <i1>
    %newSelector = init %newContinue {handshake.name = "newInit", initToken = 0 : ui1} : <i1>
    %data = mux %newSelector [%loopLiveIn, %iterLiveOut] {handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %oldContinue_buffered2 = buffer %oldContinue, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "buffer2"} : <i1>
    %newContinue2 = spec_v2_repeating_init %oldContinue_buffered2 {handshake.name = "ri2", initToken = 1 : ui1} : <i1>
    %passerOut = passer %data [%newContinue2] {handshake.name = "passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %passerOut : <i1>
  }
}
