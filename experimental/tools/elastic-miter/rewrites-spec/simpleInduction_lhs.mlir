module {
  handshake.func @simpleInduction_lhs(%a: !handshake.channel<i1>, %c: !handshake.channel<i1>, %d: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "C_in", "D_in"], resNames = ["B_out"]} {
    %a_passed = passer %a [%c] {handshake.name = "p1"} : <i1>, <i1>
    %b = passer %a_passed [%d] {handshake.name = "p2"} : <i1>, <i1>
    end {handshake.name = "end"} %b : <i1>
  }
}
