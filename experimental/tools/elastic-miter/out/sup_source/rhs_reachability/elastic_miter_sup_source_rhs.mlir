module {
  handshake.func @sup_source_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.control<> attributes {argNames = ["ctrl"], resNames = ["result"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_ctrl"} : <i1>
    %1 = ndwire %3 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_result"} : <>
    %2 = source {handshake.bb = 1 : ui32, handshake.name = "source"} : <>
    %3 = passer %2[%0] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %1 : <>
  }
}
