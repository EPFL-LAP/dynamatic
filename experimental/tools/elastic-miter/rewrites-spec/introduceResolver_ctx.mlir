module {
  handshake.func @introduceResolver_ctx(%loopContinue: !handshake.channel<i1>, %confirmSpec: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["loopContinue", "confirmSpec_backedge"], resNames = ["loopContinue", "specLoopContinue", "confirmSpec_backedge"]} {
    %lc_forked:2 = fork [2] %loopContinue {handshake.name = "ctx_fork"} : <i1>
    %confirmSpec_forked:2 = fork [2] %confirmSpec {handshake.name = "ctx_fork_cs"} : <i1>
    %lc_passer = passer %lc_forked#0 [%confirmSpec_forked#0] {handshake.name = "ctx_passer"} : <i1>, <i1>
    %lc_ri = spec_v2_nd_speculator %lc_passer {handshake.name = "ndspec"} : <i1>
    end {handshake.name = "end0"} %lc_forked#1, %lc_ri, %confirmSpec_forked#1 : <i1>, <i1>, <i1>
  }
}
