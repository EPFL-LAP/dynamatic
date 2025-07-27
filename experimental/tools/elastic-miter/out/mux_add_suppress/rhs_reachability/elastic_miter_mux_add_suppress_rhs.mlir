module {
  handshake.func @mux_add_suppress_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["ins", "sel"], resNames = ["out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_ins"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_sel"} : <i1>
    %2 = ndwire %7 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_out"} : <i1>
    %3:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "vm_fork_1"} : <i1>
    %4 = source {handshake.bb = 1 : ui32, handshake.name = "source"} : <>
    %5 = constant %4 {handshake.bb = 1 : ui32, handshake.name = "constant", value = true} : <>, <i1>
    %6 = passer %5[%3#0] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    %7 = mux %3#1 [%0, %6] {handshake.bb = 1 : ui32, handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %2 : <i1>
  }
}
