module {
  handshake.func @h_lhs(%d: !handshake.channel<i1>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["D_in", "C_in"], resNames = ["A_out", "B_out"]} {
    %t, %f = cond_br %c, %d {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i1>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i1>
    %a, %b = fork [2] %f {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %a, %b : <i1>, <i1>
  }
}
