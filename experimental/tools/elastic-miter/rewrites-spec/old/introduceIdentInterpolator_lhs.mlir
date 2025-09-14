module {
  handshake.func @introduceIdentInterpolator_lhs(%val: !handshake.channel<i1>, %a: !handshake.channel<i1>) -> (!handshake.channel<i1>) attributes {argNames = ["val", "A_in"], resNames = ["B_out"]} {
    %b = passer %a [%val] {handshake.name = "passer"} : <i1>, <i1>
    end {handshake.name = "end0"} %b : <i1>
  }
}
