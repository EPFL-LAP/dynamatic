module {
  handshake.func @sup_gamma_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_a1"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_a2"} : <i1>
    %2 = ndwire %arg2 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_c1"} : <i1>
    %3 = ndwire %arg3 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_c2"} : <i1>
    %4 = ndwire %6 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_b"} : <i1>
    %5 = mux %3 [%1, %0] {handshake.bb = 1 : ui32, handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    %6 = passer %5[%2] {handshake.bb = 1 : ui32, handshake.name = "passer"} : <i1>, <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %4 : <i1>
  }
}
