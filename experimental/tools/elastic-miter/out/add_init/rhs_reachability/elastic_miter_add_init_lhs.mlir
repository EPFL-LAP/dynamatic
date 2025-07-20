module {
  handshake.func @add_init_lhs(%arg0: !handshake.channel<i1>, ...) attributes {argNames = ["val"], resNames = []} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_val"} : <i1>
    %1 = init %0 {handshake.bb = 2 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    sink %1 {handshake.bb = 2 : ui32, handshake.name = "sink"} : <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"}
  }
}
