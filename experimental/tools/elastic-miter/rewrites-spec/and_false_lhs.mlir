module {
  handshake.func @and_false_lhs(%in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["ins"], resNames = ["out"]} {
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 0 : i1, handshake.name = "constant"} : <>, <i1>
    %out = andi %in, %cst {handshake.name = "and"} : <i1>
    end {handshake.name = "end0"} %out : <i1>
  }
}