module {
  handshake.func @a_lhs(%d: !handshake.channel<i1>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["D_in", "C_in"], resNames = ["T_out", "F_out"]} {
    %t, %f = cond_br %c, %d {handshake.bb = 1 : ui32, handshake.name = "branch"} : <i1>, <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %t, %f : <i1>, <i1>
  }
}
