module {
  handshake.func @sup_gamma_lhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, %arg2: !handshake.channel<i1>, %arg3: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["a1", "a2", "c1", "c2"], resNames = ["b"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_a1"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_a2"} : <i1>
    %2 = ndwire %arg2 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_c1"} : <i1>
    %3 = ndwire %arg3 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_c2"} : <i1>
    %4 = ndwire %13 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_b"} : <i1>
    %5:3 = fork [3] %2 {handshake.bb = 1 : ui32, handshake.name = "vm_fork_2"} : <i1>
    %6:3 = fork [3] %3 {handshake.bb = 1 : ui32, handshake.name = "vm_fork_3"} : <i1>
    %7 = passer %5#0[%6#0] {handshake.bb = 1 : ui32, handshake.name = "passer1"} : <i1>, <i1>
    %8 = not %6#1 {handshake.bb = 1 : ui32, handshake.name = "not"} : <i1>
    %9 = passer %5#1[%8] {handshake.bb = 1 : ui32, handshake.name = "passer2"} : <i1>, <i1>
    %10 = passer %6#2[%5#2] {handshake.bb = 1 : ui32, handshake.name = "passer3"} : <i1>, <i1>
    %11 = passer %0[%7] {handshake.bb = 1 : ui32, handshake.name = "passer_a1"} : <i1>, <i1>
    %12 = passer %1[%9] {handshake.bb = 1 : ui32, handshake.name = "passer_a2"} : <i1>, <i1>
    %13 = mux %10 [%12, %11] {handshake.bb = 1 : ui32, handshake.name = "mux"} : <i1>, [<i1>, <i1>] to <i1>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"} %4 : <i1>
  }
}
