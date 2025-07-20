module {
  handshake.func @introduce_ident_interpolator_rhs(%a: !handshake.channel<i1>, ...) attributes {argNames = ["A_in"], resNames = []} {
    %res = spec_v2_interpolator %a, %a {handshake.name = "interpolate"} : <i1>
    sink %res {handshake.name = "sink"} : <i1>
    end {handshake.name = "end0"}
  }
}
