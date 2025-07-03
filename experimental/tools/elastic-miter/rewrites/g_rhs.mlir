module {
  handshake.func @g_rhs(%d: !handshake.channel<i1>, %m: !handshake.channel<i1>, %n: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["D_in", "M_in", "N_in"], resNames = ["A_out"]} {
    %src = source {handshake.bb = 1 : ui32, handshake.name = "source"}: <>
    %const_1 = constant %src {value = 1 : i1, handshake.bb = 1 : ui32, handshake.name = "const"} : <>, <i1>
    %ctrl = mux %m [%n, %const_1] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %t, %f = cond_br %ctrl, %d {handshake.bb = 1 : ui32, handshake.name = "supp_br"} : <i1>, <i1>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %f : <i1>
  }
}
