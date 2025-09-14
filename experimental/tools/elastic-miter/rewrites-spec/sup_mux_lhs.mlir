module {
  handshake.func @sup_mux_lhs(%loopLiveIn: !handshake.channel<i1>, %iterLiveOut: !handshake.channel<i1>, %oldContinue: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["iterLiveIn"]} {
    %oldContinue_forked:2 = fork [2] %oldContinue {handshake.name = "fork_continue"} : <i1>
    %passerOut = passer %iterLiveOut [%oldContinue_forked#0] {handshake.name = "passer"} : <i1>, <i1>

    %oldSelector = init %oldContinue_forked#1 {handshake.name = "oldInit", initToken = 0 : ui1} : <i1>
    %data = mux %oldSelector [%loopLiveIn, %passerOut] {handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.name = "end0"} %data : <i1>
  }
}
