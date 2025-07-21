module {
  handshake.func @introduceResolver_rhs(%loopContinue: !handshake.channel<i1>, %specLoopContinue: !handshake.channel<i1>, %confirmSpec_backedge: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>) attributes {argNames = ["loopContinue", "specLoopContinue", "confirmSpec_backedge"], resNames = ["confirmSpec"]} {
    %confirmSpec = spec_v2_resolver %loopContinue, %specLoopContinue {handshake.name = "resolver"} : <i1>
    end {handshake.name = "end0"} %confirmSpec : <i1>
  }
}