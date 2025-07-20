module {
  handshake.func @add_init_lhs(%val: !handshake.channel<i1>) attributes {argNames = ["val"], resNames = []} {
    %result = init %val {handshake.bb = 1 : ui32, handshake.name = "init0", initToken = 0 : ui1} : <i1>
    sink %result {handshake.bb = 1 : ui32, handshake.name = "sink"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"}
  }
}
