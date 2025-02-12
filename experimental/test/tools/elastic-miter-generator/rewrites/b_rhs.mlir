module {
  handshake.func @b_rhs(%x: !handshake.channel<i32>, %y: !handshake.channel<i32>, %d: !handshake.channel<i32>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["X", "Y","D", "C"], resNames = ["A", "B"]} {
    %a = mux %index [%x, %y] {handshake.bb = 1 : ui32, handshake.name = "mux"}  : <i1>, <i32>
    %index = init %c ...... TODO
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %a, %d : <i32>, <i32>
  }
}