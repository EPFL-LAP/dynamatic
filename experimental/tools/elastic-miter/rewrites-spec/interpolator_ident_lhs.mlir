module {
  handshake.func @interpolator_ident_lhs(%a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %a : <i1>
  }
}
