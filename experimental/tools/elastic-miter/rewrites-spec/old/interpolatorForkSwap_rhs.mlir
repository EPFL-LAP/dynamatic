module {
  handshake.func @interpolatorForkSwap_rhs(%short: !handshake.channel<i1>, %long: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["short", "long"], resNames = ["out1", "out2"]} {
    %short_forked:2 = fork [2] %short {handshake.name = "fork_short"} : <i1>
    %long_forked:2 = fork [2] %long {handshake.name = "fork_long"} : <i1>
    %out1 = spec_v2_interpolator %short_forked#0, %long_forked#0 {handshake.name = "interpolator1"} : <i1>
    %out2 = spec_v2_interpolator %short_forked#1, %long_forked#1 {handshake.name = "interpolator2"} : <i1>
    end {handshake.name = "end0"} %out1, %out2 : <i1>, <i1>
  }
}
