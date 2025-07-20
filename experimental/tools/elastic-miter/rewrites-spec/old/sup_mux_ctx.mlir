module {
  handshake.func @sup_mux_ctx(%oldContinue: !handshake.channel<i1>, %loopLiveIn: !handshake.channel<i1>, %iterLiveOut: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["oldContinue", "loopLiveIn", "iterLiveOut"], resNames = ["oldContinue", "loopLiveIn", "iterLiveOut", "newContinue"]} {
    %oldContinueForked:2 = lazy_fork [2] %oldContinue {handshake.bb = 1 : ui32, handshake.name = "forkOldContinue"} : <i1>
    %newContinue = spec_v2_repeating_init %oldContinueForked#0 {handshake.bb = 1 : ui32, handshake.name = "ri", initToken = 1 : ui1} : <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %oldContinueForked#1, %loopLiveIn, %iterLiveOut, %newContinue : <i1>, <i1>, <i1>, <i1>
  }
}
