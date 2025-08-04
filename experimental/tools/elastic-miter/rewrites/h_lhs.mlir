module {
  handshake.func @h_lhs(%d: !handshake.channel<i32>, %c: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["D", "C"], resNames = ["A", "B"]} {
    %t, %f = cond_br %c, %d {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i32>
    sink %t {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %a, %b = fork [2] %f {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %a, %b : <i32>, <i32>
  }
}