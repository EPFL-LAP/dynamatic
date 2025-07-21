module {
  handshake.func @introduceResolver_ctx(%loopContinue: !handshake.channel<i1>, %confirmSpec: !handshake.channel<i1>, ...) -> (!handshake.channel<i1>, !handshake.channel<i1>, !handshake.channel<i1>) attributes {argNames = ["loopContinue", "confirmSpec_backedge"], resNames = ["loopContinue", "specLoopContinue", "confirmSpec_backedge"]} {
    %lc_forked:2 = fork [2] %loopContinue {handshake.name = "ctx_fork"} : <i1>
    %confirmSpec_forked:2 = fork [2] %confirmSpec {handshake.name = "ctx_fork_cs"} : <i1>
    %lc_passer = passer %lc_forked#0 [%confirmSpec_forked#0] {handshake.name = "ctx_passer"} : <i1>, <i1>
    %lc_ns = spec_v2_nd_speculator %lc_passer {handshake.name = "ndspec"} : <i1>
    %lc_ri = spec_v2_repeating_init %lc_ns {handshake.name = "ctx_ri", initToken = 1 : ui1} : <i1>
    %lc_buf = buffer %lc_ri, bufferType = FIFO_BREAK_NONE, numSlots = 1 {handshake.name = "ctx_buffer"} : <i1>
    end {handshake.name = "end0"} %lc_forked#1, %lc_buf, %confirmSpec_forked#1 : <i1>, <i1>, <i1>
  }
}
