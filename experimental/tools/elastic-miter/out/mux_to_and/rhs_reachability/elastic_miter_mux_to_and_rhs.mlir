module {
  handshake.func @mux_to_and_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in"], resNames = ["C_out"]} {
    %0 = andi %arg0, %arg1 {handshake.name = "andi1"} : <i1>
    end {handshake.name = "end0"} %0 : <i1>
  }
}
