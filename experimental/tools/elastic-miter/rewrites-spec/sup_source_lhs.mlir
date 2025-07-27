module {
  handshake.func @sup_source_lhs(%ctrl: !handshake.channel<i1>, ...) -> (!handshake.control<>) attributes {argNames = ["ctrl"], resNames = ["result"]} {
    sink %ctrl {handshake.name = "sink"} : <i1>
    %result = source {handshake.name = "source"} : <>
    end {handshake.name = "end0"} %result : <>
  }
}
