module {
  handshake.func @g_lhs(%d: !handshake.channel<i32>, %m: !handshake.channel<i1>, %n: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>) attributes {argNames = ["D", "M", "N"], resNames = ["A"]} {
    %t_0, %f_0 = cond_br %m, %d {handshake.bb = 1 : ui32, handshake.name = "supp_br_0"} : <i1>, <i32>
    sink %t_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %t_1, %f_1 = cond_br %n, %f_0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_1"} : <i1>, <i32>
    sink %t_1 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %f_1 : <i32>
  }
}