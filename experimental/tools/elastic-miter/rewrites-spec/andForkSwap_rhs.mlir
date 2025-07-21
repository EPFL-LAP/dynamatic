module {
  handshake.func @andForkSwap_rhs(%short: !handshake.channel<i1>, %long: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["in1", "in2"], resNames = ["out1", "out2"]} {
    %short_forked:2 = fork [2] %short {handshake.name = "fork1"} : <i1>
    %long_forked:2 = fork [2] %long {handshake.name = "fork2"} : <i1>
    %out1 = andi %short_forked#0, %long_forked#0 {handshake.name = "and1"} : <i1>
    %out2 = andi %short_forked#1, %long_forked#1 {handshake.name = "and2"} : <i1>
    end {handshake.name = "end0"} %out1, %out2 : <i1>, <i1>
  }
}
