module {
  handshake.func @extension3_lhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %ctrl: !handshake.channel<i1>, %sel: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["a_in", "b_in", "ctrl", "sel"], resNames = ["res"]} {
    %a_passed = passer %a [%c] {handshake.name = "passer"} : <i1>, <i1>
    %muxed = mux %sel [%b, %a_passed] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %muxed : <i1>
  }
}
