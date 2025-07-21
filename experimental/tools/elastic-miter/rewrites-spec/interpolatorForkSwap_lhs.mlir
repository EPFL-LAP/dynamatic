module {
  handshake.func @interpolatorForkSwap_lhs(%short: !handshake.channel<i1>, %long: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["short", "long"], resNames = ["out1", "out2"]} {
    %out = spec_v2_interpolator %short, %long {handshake.name = "interpolator"} : <i1>
    %out_forked:2 = fork [2] %out {handshake.name = "fork"} : <i1>
    end {handshake.name = "end0"} %out_forked#0, %out_forked#1 : <i1>, <i1>
  }
}
