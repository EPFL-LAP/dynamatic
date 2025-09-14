module {
  handshake.func @sup_mul_rhs(%a: !handshake.channel<i1>, %b: !handshake.channel<i1>, %cond: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["A_in", "B_in", "Cond"], resNames = ["C_out"]} {
    %c = muli %a, %b {handshake.bb = 1 : ui32, handshake.name = "muli"} : <i1>
    %c_passer = passer %c [%cond] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>

    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %c_passer : <i1>
  }
}
