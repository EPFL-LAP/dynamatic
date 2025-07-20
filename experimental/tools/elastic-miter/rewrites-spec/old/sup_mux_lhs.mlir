module {
  handshake.func @sup_mux_lhs(%loopLiveIn: !handshake.channel<i1>, %iterLiveOut: !handshake.channel<i1>, %oldContinue: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["iterLiveIn"]} {
    %passerOut = passer %iterLiveOut [%oldContinue] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>

    %oldSelector = init %oldContinue {handshake.bb = 1 : ui32, handshake.name = "oldInit", initToken = 0 : ui1} : <i1>
    %data = mux %oldSelector [%loopLiveIn, %passerOut] {handshake.bb = 1 : ui32, handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %data : <i1>
  }
}
