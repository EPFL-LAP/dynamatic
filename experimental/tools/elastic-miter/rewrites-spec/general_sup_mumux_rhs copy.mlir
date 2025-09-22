module {
  handshake.func @general_sup_mumux_rhs(%loopLiveIn: !handshake.channel<i1>, %iterLiveOut: !handshake.channel<i1>, %oldContinue: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue", "ctrl"], resNames = ["iterLiveIn"]} {
    %ctrl_not = not %ctrl {handshake.name = "ctrl_not"} : <i1>
    %ori = ori %oldContinue, %ctrl_not {handshake.name = "ori"} : <i1>
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 1 : i1, handshake.name = "constant"} : <>, <i1>
    %newContinue = mux %newSelector2 [%cst, %ori] {handshake.name = "ctrl_mux"} : <i1> , [<i1>, <i1>] to <i1>
    %newContinue_forked:3 = fork [3] %newContinue {handshake.name = "fork_continue"} : <i1>
    %newSelector = init %newContinue_forked#0 {handshake.name = "newInit", initToken = 0 : ui1} : <i1>
    %newContinue_buffered = buffer %newContinue_forked#1, bufferType = ONE_SLOT_BREAK_DV, numSlots = 1 {debugCounter = false, handshake.name = "buff"} : <i1>
    %newSelector2 = init %newContinue_buffered {handshake.name = "newInit2", initToken = 0 : ui1} : <i1>
    %data = mux %newSelector [%loopLiveIn, %iterLiveOut] {handshake.name = "data_mux"}  : <i1>, [<i1>, <i1>] to <i1>

    %passerOut = passer %data [%newContinue_forked#2] {handshake.name = "passer"} : <i1>, <i1>

    end {handshake.name = "end0"} %passerOut : <i1>
  }
}
