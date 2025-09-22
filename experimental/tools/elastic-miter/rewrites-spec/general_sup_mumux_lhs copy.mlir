module {
  handshake.func @general_sup_mumux_lhs(%loopLiveIn: !handshake.channel<i1>, %iterLiveOut: !handshake.channel<i1>, %oldContinue: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue", "ctrl"], resNames = ["iterLiveIn"]} {
    %oldContinue_forked:2 = fork [2] %oldContinue {handshake.name = "fork_continue"} : <i1>
    %ctrl_forked:2 = fork [2] %ctrl {handshake.name = "ctrl_fork"} : <i1>
    %andi = andi %oldContinue_forked#0, %ctrl_forked#0 {handshake.name = "andi"} : <i1>
    %passerOut = passer %iterLiveOut [%andi] {handshake.name = "passer"} : <i1>, <i1>

    %sel_passed = passer %oldContinue_forked#1 [%ctrl_forked#1] {handshake.name = "sel_passer"} : <i1>, <i1>

    %oldSelector = init %sel_passed {handshake.name = "oldInit", initToken = 0 : ui1} : <i1>
    %data = mux %oldSelector [%loopLiveIn, %passerOut] {handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    end {handshake.name = "end0"} %data : <i1>
  }
}
