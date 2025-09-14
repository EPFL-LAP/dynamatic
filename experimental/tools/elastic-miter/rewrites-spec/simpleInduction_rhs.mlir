module {
  handshake.func @simpleInduction_rhs(%a: !handshake.channel<i1>, %c: !handshake.channel<i1>, %d: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "C_in", "D_in"], resNames = ["B_out"]} {
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 0 : i1, handshake.name = "constant"} : <>, <i1>
    %muxed = mux %c [%cst, %d] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %b = passer %a [%muxed] {handshake.name = "p1"} : <i1>, <i1>
    end {handshake.name = "end"} %b : <i1>
  }
}
