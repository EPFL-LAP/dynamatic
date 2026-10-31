module {
  handshake.func @g_rhs(%D: <i32>, %M: <i1>, %N: <i1>) -> (A: <i32>) {
    %src = source {handshake.bb = 1 : ui32, handshake.name = "source"}: <>
    %const_1 = constant %src {value = 1 : i1, handshake.bb = 1 : ui32, handshake.name = "const"} : <>, <i1>
    %ctrl = mux %M [%N, %const_1] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, [<i1>, <i1>] to <i1>
    %t, %f = cond_br %ctrl, %D {handshake.bb = 1 : ui32, handshake.name = "supp_br"} : <i1>, <i32>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink"} : <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %f : <i32>
  }
}