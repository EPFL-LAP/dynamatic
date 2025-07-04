module {
  handshake.func @interpolator_ident_lhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %0 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %1 : <i1>
  }
}
