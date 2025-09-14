module {
  handshake.func @muxForkSwap_rhs(%sel: !handshake.channel<i1>, %data: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["sel_in", "data_in"], resNames = ["out1", "out2"]} {
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 0 : i1, handshake.name = "constant"} : <>, <i1>
    %out = mux %sel [%cst, %data] {handshake.name = "mux0"} : <i1>, [<i1>, <i1>] to <i1>
    %out_forked:2 = fork [2] %out {handshake.name = "fork"} : <i1>
    end {handshake.name = "end0"} %out_forked#0, %out_forked#1 : <i1>, <i1>
  }
}
