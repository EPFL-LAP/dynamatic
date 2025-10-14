module {
  handshake.func @mux_to_and_rhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "B_in"], resNames = ["C_out"]} {
    %out = andi %a, %b {handshake.name = "andi1"} : <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}
