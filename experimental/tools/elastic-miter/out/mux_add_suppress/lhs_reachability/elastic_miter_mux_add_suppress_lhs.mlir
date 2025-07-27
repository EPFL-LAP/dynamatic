module {
  handshake.func @mux_add_suppress_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["ins", "sel"], resNames = ["out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_ins"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_sel"} : <i1>
    %2 = ndwire %5 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_out"} : <i1>
    %3 = source {handshake.bb = 1 : ui32, handshake.name = "source"} : <>
    %4 = constant %3 {handshake.bb = 1 : ui32, handshake.name = "constant", value = true} : <>, <i1>
    %5 = mux %1 [%0, %4] {handshake.bb = 1 : ui32, handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %2 : <i1>
  }
}
