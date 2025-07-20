module {
  handshake.func @introduce_ident_interpolator_lhs(%arg0: !handshake.channel<i1>, ...) attributes {argNames = ["A_in"], resNames = []} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    sink %0 {handshake.bb = 1 : ui32, handshake.name = "vm_sink_0"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"}
  }
}
