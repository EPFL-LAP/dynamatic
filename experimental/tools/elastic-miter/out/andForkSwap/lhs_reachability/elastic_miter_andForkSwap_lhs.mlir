module {
  handshake.func @andForkSwap_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["in2", "in1"], resNames = ["out1", "out2"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_in2"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_in1"} : <i1>
    %2 = ndwire %5#0 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_out1"} : <i1>
    %3 = ndwire %5#1 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_out2"} : <i1>
    %4 = andi %1, %0 {handshake.bb = 1 : ui32, handshake.name = "and"} : <i1>
    %5:2 = fork [2] %4 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %2, %3 : <i1>, <i1>
  }
}
