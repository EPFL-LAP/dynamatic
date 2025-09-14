module {
  handshake.func @sup_mux_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["iterLiveIn"]} {
    %0:2 = fork [2] %arg2 {handshake.name = "fork_continue"} : <i1>
    %1 = passer %arg1[%0#0] {handshake.name = "passer"} : <i1>, <i1>
    %2 = init %0#1 {handshake.name = "oldInit", initToken = 0 : ui1} : <i1>
    %3 = mux %2 [%arg0, %1] {handshake.name = "data_mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %3 : <i1>
  }
}
