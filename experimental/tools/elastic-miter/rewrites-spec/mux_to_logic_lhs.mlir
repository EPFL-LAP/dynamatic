module {
  handshake.func @mux_to_logic_lhs(%in1: !handshake.channel<i1>, %in2: !handshake.channel<i1>, %sel: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["in1", "in2", "sel"], resNames = ["out"]} {
    %sel_not = not %sel {handshake.name = "not"} : <i1>
    %in1_passer = passer %in1 [%sel] {handshake.name = "passer1"} : <i1>, <i1>
    %in2_passer = passer %in2 [%sel_not] {handshake.name = "passer2"} : <i1>, <i1>
    %out = mux %sel [%in2_passer, %in1_passer] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}
