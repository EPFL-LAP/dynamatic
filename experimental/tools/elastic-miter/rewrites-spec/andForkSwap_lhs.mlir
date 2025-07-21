module {
  handshake.func @andForkSwap_lhs(%short: !handshake.channel<i1>, %long: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["in1", "in2"], resNames = ["out1", "out2"]} {
    %out = andi %short, %long {handshake.name = "and"} : <i1>
    %out_forked:2 = fork [2] %out {handshake.name = "fork"} : <i1>
    end {handshake.name = "end0"} %out_forked#0, %out_forked#1 : <i1>, <i1>
  }
}
