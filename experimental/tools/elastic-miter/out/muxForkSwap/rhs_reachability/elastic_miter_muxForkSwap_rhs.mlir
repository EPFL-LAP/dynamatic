module {
  handshake.func @muxForkSwap_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["sel_in", "data_in"], resNames = ["out1", "out2"]} {
    %0 = source {handshake.name = "source"} : <>
    %1 = constant %0 {handshake.name = "constant", value = false} : <>, <i1>
    %2 = mux %arg0 [%1, %arg1] {handshake.name = "mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %3:2 = fork [2] %2 {handshake.name = "fork"} : <i1>
    end {handshake.name = "end0"} %3#0, %3#1 : <i1>, <i1>
  }
}
