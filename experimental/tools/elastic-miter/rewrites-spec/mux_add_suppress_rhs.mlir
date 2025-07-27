module {
  handshake.func @mux_add_suppress_rhs(%in: !handshake.channel<i1>, %sel: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["ins", "sel"], resNames = ["out"]} {
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 1 : i1, handshake.name = "constant"} : <>, <i1>
    %cst_passer = passer %cst [%sel] {handshake.name = "passer"} : <i1>, <i1>
    %out = mux %sel [%in, %cst_passer] {handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}
