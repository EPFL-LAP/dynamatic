module {
  handshake.func @muxForkSwap_lhs(%sel: !handshake.channel<i1>, %data: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["sel_in", "data_in"], resNames = ["out1", "out2"]} {
    %sel_forked:2 = fork [2] %sel {handshake.name = "sel_fork"} : <i1>
    %data_forked:2 = fork [2] %data {handshake.name = "data_fork"} : <i1>
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 0 : i1, handshake.name = "constant"} : <>, <i1>
    %out1 = mux %sel_forked#0 [%cst, %data_forked#0] {handshake.name = "mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %src2 = source {handshake.name = "source2"} : <>
    %cst2 = constant %src2 {value = 0 : i1, handshake.name = "constant2"} : <>, <i1>
    %out2 = mux %sel_forked#1 [%cst2, %data_forked#1] {handshake.name = "mux1"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %out1, %out2 : <i1>, <i1>
  }
}
