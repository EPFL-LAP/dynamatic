module {
  handshake.func @resolver_rhs(%arg0: !handshake.channel<i1>, %arg1: !handshake.channel<i1>, ...) -> !handshake.channel<i1> attributes {argNames = ["Actual", "Generated"], resNames = ["Confirm"]} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Actual"} : <i1>
    %1 = ndwire %arg1 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_Generated"} : <i1>
    %2 = ndwire %3 {handshake.bb = 3 : ui32, handshake.name = "ndw_out_Confirm"} : <i1>
    %3 = spec_v2_resolver %0, %1 {handshake.bb = 1 : ui32, handshake.name = "resolver"} : <i1>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"} %2 : <i1>
  }
}
