module {
  handshake.func @ri_fork_lhs(%arg0: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["A_in"], resNames = ["B_out", "C_out"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <i1>
    %1 = ndwire %4#0 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_B_out"} : <i1>
    %2 = ndwire %4#1 {handshake.bb = 2 : ui32, handshake.name = "ndw_out_C_out"} : <i1>
    %3 = spec_v2_repeating_init %0 {handshake.bb = 1 : ui32, handshake.name = "ri", initToken = 1 : ui1} : <i1>
    %4:2 = fork [2] %3 {handshake.bb = 1 : ui32, handshake.name = "fork"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %1, %2 : <i1>, <i1>
  }
}
