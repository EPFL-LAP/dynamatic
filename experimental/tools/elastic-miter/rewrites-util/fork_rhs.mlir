module {
  handshake.func @fork_rhs(%a: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %result:2 = lazy_fork [2] %a {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %result#0, %result#1 : <i1>, <i1>
  }
}
