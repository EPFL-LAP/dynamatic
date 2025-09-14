module {
  handshake.func @muxForkSwap_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["sel_in", "data_in"], resNames = ["out1", "out2"]} {
    %0:2 = fork [2] %arg0 {handshake.name = "sel_fork"} : <i1>
    %1:2 = fork [2] %arg1 {handshake.name = "data_fork"} : <i1>
    %2 = source {handshake.name = "source"} : <>
    %3 = constant %2 {handshake.name = "constant", value = false} : <>, <i1>
    %4 = mux %0#0 [%3, %1#0] {handshake.name = "mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %5 = source {handshake.name = "source2"} : <>
    %6 = constant %5 {handshake.name = "constant2", value = false} : <>, <i1>
    %7 = mux %0#1 [%6, %1#1] {handshake.name = "mux1"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %4, %7 : <i1>, <i1>
  }
}
