module {
  handshake.func @test_lhs(%arg0: !handshake.control<>, ...) attributes {argNames = ["A_in"], resNames = []} {
    %0 = ndwire %arg0 {handshake.bb = 0 : ui32, handshake.name = "ndw_in_A_in"} : <>
    %1 = ndsource {handshake.bb = 1 : ui32, handshake.name = "nds"} : <>
    %2 = join %0, %1 {handshake.bb = 1 : ui32, handshake.name = "join"} : <>
    sink %2 {handshake.bb = 1 : ui32, handshake.name = "sink"} : <>
    end {handshake.bb = 2 : ui32, handshake.name = "end0"}
  }
}
