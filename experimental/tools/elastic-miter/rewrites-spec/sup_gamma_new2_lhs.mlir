module {
  handshake.func @sup_gamma_new2_lhs(%a1: !handshake.channel<i1>, %a2: !handshake.channel<i1>, %c1: !handshake.channel<i1>, %c2: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %c1_forked:3 = fork [3] %c1 {handshake.name = "c1_fork"} : <i1>
    %c2_passed = passer %c2 [%c1_forked#0] {handshake.name = "c2_passer"} : <i1>, <i1>
    %c2_forked:3 = fork [3] %c2_passed {handshake.name = "c2_fork"} : <i1>
    %c2_not = not %c2_forked#2 {handshake.name = "not"} : <i1>
    %a1_passed1 = passer %a1 [%c1_forked#1] {handshake.name = "a1_passer1"} : <i1>, <i1>
    %a1_passed2 = passer %a1_passed1 [%c2_forked#0] {handshake.name = "a1_passer2"} : <i1>, <i1>
    %a2_passed1 = passer %a2 [%c1_forked#2] {handshake.name = "a2_passer1"} : <i1>, <i1>
    %a2_passed2 = passer %a2_passed1 [%c2_not] {handshake.name = "a2_passer2"} : <i1>, <i1>
    %b = mux %c2_forked#1 [%a2_passed2, %a1_passed2] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %b : <i1>
  }
}
