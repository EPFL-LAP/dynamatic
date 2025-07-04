module {
  handshake.func @interpolator_ind_lhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %sup_in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "B_in", "Sup_in"], resNames = ["Sup_out"]} {
    %b_forked:2 = fork [2] %b {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    %interpolate = spec_v2_interpolator %a, %b_forked#0 {handshake.bb = 1 : ui32, handshake.name = "interpolate"} : <i1>
    %ri = spec_v2_repeating_init %b_forked#1 {handshake.bb = 1 : ui32, handshake.name = "ri"} : <i1>
    %passer1 = passer %sup_in [%ri] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %passer2 = passer %passer1 [%interpolate] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %passer2 : <i1>
  }
}
