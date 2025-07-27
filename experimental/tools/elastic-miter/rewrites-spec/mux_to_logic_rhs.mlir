module {
  handshake.func @mux_to_logic_rhs(%in1: !handshake.channel<i1>, %in2: !handshake.channel<i1>, %sel: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["in1", "in2", "sel"], resNames = ["out"]} {
    %sel_not = not %sel {handshake.name = "not"} : <i1>
    %in1_and = andi %in1, %sel {handshake.name = "and1"} : <i1>
    %in2_and = andi %in2, %sel_not {handshake.name = "and2"} : <i1>
    %out = ori %in1_and, %in2_and {handshake.name = "or"} : <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}
