module {
  handshake.func @introduce_ident_interpolator_rhs(%arg0: !handshake.channel<i1>, ...) attributes {argNames = ["A_in"], resNames = []} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1:2 = fork [2] %0 {handshake.bb = 1 : ui32, handshake.name = "vm_fork_0"} : <i1>
    %2 = spec_v2_interpolator %1#0, %1#1 {handshake.bb = 1 : ui32, handshake.name = "interpolate"} : <i1>
    sink %2 {handshake.bb = 1 : ui32, handshake.name = "sink"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"}
  }
}
