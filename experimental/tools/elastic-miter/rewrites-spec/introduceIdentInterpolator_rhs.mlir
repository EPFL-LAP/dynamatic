module {
  handshake.func @introduceIdentInterpolator_lhs(%a: !handshake.channel<i1>) -> (!handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %a_forked:2 = fork [2] %a {handshake.name = "a_fork"} : <i1>
    %b = spec_v2_interpolator %a_forked#0, %a_forked#1 {handshake.name = "interpolate"} : <i1>
    end {handshake.name = "end0"} %b : <i1>
  }
}
