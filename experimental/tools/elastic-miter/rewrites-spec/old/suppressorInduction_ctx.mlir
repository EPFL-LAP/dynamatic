module {
  handshake.func @suppressorInduction_rhs(%short: !handshake.channel<i1>, %a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["short", "A_in"], resNames = ["short", "oldLong", "A_in"]} {
    %short_forked:2 = fork [2] %short {handshake.name = "fork_short"} : <i1>
    %oldLong = spec_v2_nd_speculator %short_forked#0 {handshake.name = "nd_spec"} : <i1>
    end {handshake.name = "end"} %short_forked#1, %oldLong, %a : <i1>, <i1>, <i1>
  }
}
