module {
  handshake.func @repeating_init_lhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %0 = spec_v2_repeating_init %arg0 {handshake.name = "spec_v2_repeating_init", initToken = 1 : ui1} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %0 : <i1>
  }
}
