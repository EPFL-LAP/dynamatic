module {
  handshake.func @sup_mux_rhs(%loopLiveIn: !handshake.channel<i1>, %newContinue: !handshake.channel<i1>, %iterLiveOut: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopLiveIn", "newContinue", "iterLiveOut"], resNames = ["iterLiveIn"]} {
    %newSelector = init %newContinue {handshake.bb = 1 : ui32, handshake.name = "newInit", initToken = 0 : ui1} : <i1>
    %data = mux %newSelector [%loopLiveIn, %iterLiveOut] {handshake.bb = 1 : ui32, handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %passerOut = passer %data [%newContinue] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %passerOut : <i1>
  }
}
