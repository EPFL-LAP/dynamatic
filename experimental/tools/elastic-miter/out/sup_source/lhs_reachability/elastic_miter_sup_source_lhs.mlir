module {
  handshake.func @sup_source_lhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.control<> attributes {argNames = ["ctrl"], resNames = ["result"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_ctrl"} : <i1>
    %1 = ndwire %2 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_result"} : <>
    sink %0 {handshake.bb = 1 : ui32, handshake.name = "sink"} : <i1>
    %2 = source {handshake.bb = 1 : ui32, handshake.name = "source"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %1 : <>
  }
}
