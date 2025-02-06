module {
  handshake.func @a_lhs(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["D", "C"], resNames = ["T", "F"]} {
    %t, %f = cond_br %c, %d {handshake.bb = 1 : ui32, handshake.name = "branch"} : <i1>, <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %t, %f : <i32>, <i32>
  }
}