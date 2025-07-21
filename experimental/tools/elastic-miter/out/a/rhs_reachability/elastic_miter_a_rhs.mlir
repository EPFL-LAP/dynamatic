module {
  handshake.func @a_rhs(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["D", "C"], resNames = ["T", "F"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_D"} : <i32>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_C"} : <i1>
    %2 = ndwire %falseResult {handshake.bb = 3 : ui32, handshake.name = "ndw_out_T"} : <i32>
    %3 = ndwire %falseResult_1 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_F"} : <i32>
    %4:2 = fork [2] %0 {handshake.bb = 1 : ui32, handshake.name = "fork_data"} : <i32>
    %5:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "fork_control"} : <i1>
    %6 = not %5#0 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %trueResult, %falseResult = cond_br %6, %4#0 {handshake.bb = 1 : ui32, handshake.name = "supp_br_T"} : <i1>, <i32>
    sink %trueResult {handshake.bb = 1 : ui32, handshake.name = "supp_sink_0"} : <i32>
    %trueResult_0, %falseResult_1 = cond_br %5#1, %4#1 {handshake.bb = 1 : ui32, handshake.name = "supp_br_F"} : <i1>, <i32>
    sink %trueResult_0 {handshake.bb = 1 : ui32, handshake.name = "supp_sink_1"} : <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end0"} %2, %3 : <i32>, <i32>
  }
}
