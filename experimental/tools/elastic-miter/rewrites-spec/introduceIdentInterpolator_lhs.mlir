module {
  handshake.func @introduceIdentInterpolator_lhs(%a: !handshake.channel<i1>) -> (!handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out"]} {
    %a_not = not %a {handshake.name = "not1"} : <i1>
    %b = not %a_not {handshake.name = "not2"} : <i1>
    end {handshake.name = "end0"} %b : <i1>
  }
}
