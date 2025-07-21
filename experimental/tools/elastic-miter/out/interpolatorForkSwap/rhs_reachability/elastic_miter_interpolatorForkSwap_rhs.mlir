module {
  handshake.func @interpolatorForkSwap_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["long", "short"], resNames = ["out1", "out2"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_long"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_short"} : <i1>
    %2 = ndwire %6 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_out1"} : <i1>
    %3 = ndwire %7 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_out2"} : <i1>
    %4:2 = fork [2] %1 {handshake.bb = 1 : ui32, handshake.name = "fork_short"} : <i1>
    %5:2 = fork [2] %0 {handshake.bb = 1 : ui32, handshake.name = "fork_long"} : <i1>
    %6 = spec_v2_interpolator %4#0, %5#0 {handshake.bb = 1 : ui32, handshake.name = "interpolator1"} : <i1>
    %7 = spec_v2_interpolator %4#1, %5#1 {handshake.bb = 1 : ui32, handshake.name = "interpolator2"} : <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %2, %3 : <i1>, <i1>
  }
}
