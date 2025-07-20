module {
  handshake.func @introduceIdentInterpolator_rhs(%val: !handshake.channel<i1>, %a: !handshake.channel<i1>) -> (!handshake.channel<i1>) attributes {argNames = ["val", "A_in"], resNames = ["B_out"]} {
    %val2 = spec_v2_interpolator %val, %val {handshake.name = "interpolate"} : <i1>
    %b = passer %a [%val2] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %b : <i1>
  }
}
