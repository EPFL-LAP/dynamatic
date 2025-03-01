module {
  handshake.func @d_rhs(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>) attributes {argNames = ["D", "C"], resNames = ["A"]} {
    sink %c {handshake.bb = 1 : ui32, handshake.name = "sink"} : <i1>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %d : <i32>
  }
}