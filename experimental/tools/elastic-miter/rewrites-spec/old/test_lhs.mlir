module {
  handshake.func @test_lhs(%a: !handshake.control<>) attributes {argNames = ["A_in"], resNames = []} {
    %src = ndsource {handshake.name = "nds"} : <>
    %b = join %a, %src {handshake.name = "join"} : <>
    sink %b {handshake.name = "sink"} : <>
    end {handshake.bb = 1 : ui32, handshake.name = "end0"}
  }
}
