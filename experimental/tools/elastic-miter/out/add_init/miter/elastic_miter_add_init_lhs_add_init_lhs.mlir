module {
  handshake.func @elastic_miter_add_init_lhs_add_init_lhs(%arg0: !handshake.channel<i1>, ...) attributes {argNames = ["val"], resNames = []} {
    %0:2 = lazy_fork [2] %arg0 {handshake.bb = 0 : ui32, handshake.name = "in_fork_val"} : <i1>
    %1 = buffer %0#0, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_buf_val"} : <i1>
    %2 = buffer %0#1, bufferType = FIFO_BREAK_DV, numSlots = 1 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_buf_val"} : <i1>
    %3 = ndwire %1 {handshake.bb = 0 : ui32, handshake.name = "lhs_in_ndw_val"} : <i1>
    %4 = ndwire %2 {handshake.bb = 0 : ui32, handshake.name = "rhs_in_ndw_val"} : <i1>
    sink %3 {handshake.bb = 1 : ui32, handshake.name = "lhs_vm_sink_0"} : <i1>
    %5 = init %4 {handshake.bb = 4 : ui32, handshake.name = "rhs_init0", initToken = 0 : ui1} : <i1>
    sink %5 {handshake.bb = 4 : ui32, handshake.name = "rhs_sink"} : <i1>
    end {handshake.bb = 5 : ui32, handshake.name = "end"}
  }
}
