module {
  handshake.func @sup_source_rhs(%ctrl: !handshake.channel<i1>, ...) -> (!handshake.control<>) attributes {argNames = ["ctrl"], resNames = ["result"]} {
    %result = source {handshake.name = "source"} : <>
    %result_passer = passer %result [%ctrl] {handshake.name = "passer"} : <>, <i1>
    end {handshake.name = "end0"} %result_passer : <>
  }
}
