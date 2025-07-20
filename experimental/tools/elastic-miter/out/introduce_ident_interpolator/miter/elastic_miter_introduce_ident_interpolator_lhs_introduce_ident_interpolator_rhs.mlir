module {
  handshake.func @elastic_miter_introduce_ident_interpolator_lhs_introduce_ident_interpolator_rhs(%arg0: !handshake.channel<i1>, ...) attributes {argNames = ["A_in"], resNames = []} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_A_in"} : <i1>
    %1 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_A_in"} : <i1>
    %2 = ndwire %0#0 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_A_in"} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_A_in"} : <i1>
    sink %2 {handshake.bb = 1 : ui32, handshake.name = "lhs_vm_sink_0"} : <i1>
    %4:2 = fork [2] %3 {handshake.bb = 2 : ui32, handshake.name = "rhs_vm_fork_0"} : <i1>
    %5 = spec_v2_interpolator %4#0, %4#1 {handshake.bb = 2 : ui32, handshake.name = "rhs_interpolate"} : <i1>
    sink %5 {handshake.bb = 2 : ui32, handshake.name = "rhs_sink"} : <i1>
    end {handshake.bb = 3 : ui32, handshake.name = "end"}
  }
}
