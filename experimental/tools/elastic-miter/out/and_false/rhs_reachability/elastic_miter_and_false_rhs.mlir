module {
  handshake.func @and_false_rhs(%arg0: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["ins"], resNames = ["out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_ins"} : <i1>
    %1 = ndwire %3 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_out"} : <i1>
    sink %0 {handshake.bb = 1 : ui32, handshake.name = "sink"} : <i1>
    %2 = source {handshake.bb = 1 : ui32, handshake.name = "source"} : <>
    %3 = constant %2 {handshake.bb = 1 : ui32, handshake.name = "constant", value = false} : <>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %1 : <i1>
  }
}
