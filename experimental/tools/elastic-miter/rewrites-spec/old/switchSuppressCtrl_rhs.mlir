module {
  handshake.func @switchSuppressCtrl_rhs(%val2: !handshake.channel<i1>, %a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["val2", "A_in"], resNames = ["B_out"]} {
    %b = passer %a [%val2] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %b : <i1>
  }
}
