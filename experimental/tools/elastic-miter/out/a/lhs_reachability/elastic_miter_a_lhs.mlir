module {
  handshake.func @a_lhs(%arg0: !handshake.channel<i32>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i32>, !handshake.channel<i32>) attributes {argNames = ["D", "C"], resNames = ["T", "F"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_D"} : <i32>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_C"} : <i1>
    %2 = ndwire %trueResult {handshake.bb = 3 : ui32, handshake.name = "ndw_out_T"} : <i32>
    %3 = ndwire %falseResult {handshake.bb = 3 : ui32, handshake.name = "ndw_out_F"} : <i32>
    %trueResult, %falseResult = cond_br %1, %0 {handshake.bb = 1 : ui32, handshake.name = "branch"} : <i1>, <i32>
    end {handshake.bb = 4 : ui32, handshake.name = "end"} %2, %3 : <i32>, <i32>
  }
}
