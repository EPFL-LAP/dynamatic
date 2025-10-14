module {
  handshake.func @mux_to_and_lhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "B_in"], resNames = ["C_out"]} {
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 0 : i1, handshake.name = "constant"} : <>, <i1>
    %a_forked:2 = fork [2] %a {handshake.name = "a_fork"} : <i1>
    %b_passed = passer %b [%a_forked#0] {handshake.name = "passer1"} : <i1>, <i1>
    %out = mux %a_forked#1 [%cst, %b_passed] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}
