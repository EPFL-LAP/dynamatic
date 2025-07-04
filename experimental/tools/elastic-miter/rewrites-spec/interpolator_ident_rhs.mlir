module {
  handshake.func @interpolator_ident_rhs(%a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %a_forked:2 = fork [2] %a {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    %a_interpolate = spec_v2_interpolator %a_forked#0, %a_forked#1 {handshake.bb = 1 : ui32, handshake.name = "interpolate"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %a_interpolate : <i1>
  }
}
