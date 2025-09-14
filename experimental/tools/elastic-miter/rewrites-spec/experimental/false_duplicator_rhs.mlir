module {
  handshake.func @false_duplicator_rhs(%in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["ins"], resNames = ["out"]} {
    %a = specv2_false_duplicator %in {handshake.name = "fd1"} : <i1>
    %b = specv2_false_duplicator %a {handshake.name = "fd2"} : <i1>
    %c = specv2_false_duplicator %b {handshake.name = "fd3"} : <i1>
    end {handshake.name = "end0"} %c : <i1>
  }
}