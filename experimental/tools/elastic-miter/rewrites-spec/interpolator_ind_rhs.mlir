module {
  handshake.func @interpolator_ind_rhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %sup_in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "B_in", "Sup_in"], resNames = ["Sup_out"]} {
    %ri = spec_v2_repeating_init %b {handshake.bb = 1 : ui32, handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %interpolate = spec_v2_interpolator %a, %ri {handshake.bb = 1 : ui32, handshake.name = "interpolate"} : <i1>
    %sup_out = passer %sup_in [%interpolate] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %sup_out : <i1>
  }
}
