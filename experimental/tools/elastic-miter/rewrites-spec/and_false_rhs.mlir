module {
  handshake.func @and_false_rhs(%in: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["ins"], resNames = ["out"]} {
    sink %in {handshake.name = "sink"} : <i1>
    %src = source {handshake.name = "source"} : <>
    %cst = constant %src {value = 0 : i1, handshake.name = "constant"} : <>, <i1>
    end {handshake.name = "end0"} %cst : <i1>
  }
}