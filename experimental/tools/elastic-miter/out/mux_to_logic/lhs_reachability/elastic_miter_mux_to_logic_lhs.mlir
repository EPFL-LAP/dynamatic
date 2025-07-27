module {
  handshake.func @mux_to_logic_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["in1", "in2", "sel"], resNames = ["out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_in1"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_in2"} : <i1>
    %2 = ndwire %arg2 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_sel"} : <i1>
    %3 = ndwire %8 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_out"} : <i1>
    %4:3 = fork [3] %2 {handshake.bb = 1 : ui32, handshake.name = "vm_fork_2"} : <i1>
    %5 = not %4#0 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %6 = passer %0[%4#1] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %7 = passer %1[%5] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    %8 = mux %4#2 [%7, %6] {handshake.bb = 1 : ui32, handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %3 : <i1>
  }
}
