module {
  handshake.func @sup_mul_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["A_in", "B_in", "Cond"], resNames = ["C_out"]} {
    %0:2 = fork [2] %arg2 {handshake.bb = 1 : ui32, handshake.name = "fork_cond"} : <i1>
    %1 = passer %arg0[%0#0] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %2 = passer %arg1[%0#1] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    %3 = muli %1, %2 {handshake.bb = 1 : ui32, handshake.name = "muli"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %3 : <i1>
  }
}
