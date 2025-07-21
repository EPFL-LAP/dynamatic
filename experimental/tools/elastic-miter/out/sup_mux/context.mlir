module {
  handshake.func @context(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["loopLiveIn", "iterLiveOut", "oldContinue"], resNames = ["loopLiveIn", "iterLiveOut", "oldContinue"]} {
    end {handshake.bb = 1 : ui32, handshake.name = "end"} %arg0, %arg1, %arg2 : <i1>, <i1>, <i1>
  }
}
